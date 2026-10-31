"""The web app: ask where something is, manage items and zones, see the latest scan."""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ask import item_in_question
from .camera import Camera
from .detector import Detector, clean_box
from .places import ago, describe, place_of
from .scanner import Scanner
from .store import Store
from .stream import LiveSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DATA_DIR = os.environ.get("DATA_DIR", "./data")
SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
CAMERA = os.environ.get("CAMERA", "/dev/video0")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "512"))

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
store = Store(os.path.join(DATA_DIR, "vlm_on_pi.db"))
# BOX_ZOOM=0 skips the zoom-in pass: faster looks, rougher boxes.
detector = Detector(os.environ.get("LLM_URL", "http://127.0.0.1:8080"),
                    zoom=os.environ.get("BOX_ZOOM", "1") != "0")
# A folder of photos has no live view; every other source streams.
live = None if CAMERA.startswith("folder:") else LiveSource(
    CAMERA, width=int(os.environ.get("LIVE_WIDTH", "640")), fps=int(os.environ.get("LIVE_FPS", "10")),
    look_size=IMAGE_SIZE)
scanner = Scanner(
    store,
    live or Camera(CAMERA, size=IMAGE_SIZE),
    detector,
    SNAPSHOT_DIR,
    every_s=int(os.environ.get("SCAN_EVERY_S", "120")),
    keep_days=int(os.environ.get("KEEP_DAYS", "7")),
)


@asynccontextmanager
async def lifespan(_app):
    if live:
        live.start()
    scanner.start()
    yield
    scanner.stop()
    if live:
        live.stop()


app = FastAPI(title="VLM on Pi", lifespan=lifespan)


class NewItem(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class NewZone(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    bbox: list[float] = Field(min_length=4, max_length=4)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/ask")
def ask(q: str):
    items = [i["name"] for i in store.items()]
    item = item_in_question(q, items)
    if item is None:
        names = ", ".join(items) or "nothing yet"
        return {"answer": "I'm not tracking that. I'm looking for: %s." % names, "sighting": None}
    sighting = store.last_sighting(item)
    if sighting is None:
        return {"answer": "I haven't seen the %s yet." % item, "sighting": None}
    return {"answer": describe(sighting, store.zones()), "sighting": sighting}


@app.get("/api/items")
def list_items():
    zones = store.zones()
    seen = {s["item"]: s for s in store.last_sightings()}
    last = store.last_scan()
    # In view = the latest look said yes; None when that look didn't ask about the item.
    present = (last or {}).get("present") or {}
    now = time.time()
    out = []
    for item in store.items():
        s = seen.get(item["name"])
        out.append(dict(item, sighting=s,
                        in_view=present.get(item["name"]),
                        where=place_of(s["bbox"], zones) if s else None,
                        when=ago(now - s["seen_at"]) if s else None))
    return out


@app.post("/api/items", status_code=201)
def add_item(body: NewItem):
    try:
        item = store.add_item(body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    scanner.scan_now()  # look for the new item straight away
    return item


@app.delete("/api/items/{item_id}", status_code=204)
def remove_item(item_id: int):
    if not store.remove_item(item_id):
        raise HTTPException(404, "No such item.")


@app.get("/api/zones")
def list_zones():
    return store.zones()


@app.post("/api/zones", status_code=201)
def add_zone(body: NewZone):
    box = clean_box(body.bbox)
    if box is None:
        raise HTTPException(400, "Draw a box first.")
    try:
        return store.add_zone(body.name, box)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/zones/{zone_id}", status_code=204)
def remove_zone(zone_id: int):
    if not store.remove_zone(zone_id):
        raise HTTPException(404, "No such zone.")


@app.get("/api/status")
def status():
    scan = store.last_scan()
    if scan:
        scan.pop("raw", None)
        scan["when"] = ago(time.time() - scan["started_at"])
    next_in = None
    if scanner.last_started and not scanner.busy:
        next_in = max(0, int(scanner.last_started + scanner.every_s - time.time()))
    return {"scanning": scanner.busy, "model_ready": detector.healthy(), "every_s": scanner.every_s,
            "next_look_in": next_in, "last_scan": scan,
            "camera": {"source": os.path.basename(CAMERA.rstrip("/")) or CAMERA,
                       "live": bool(live and live.live), "error": live.error if live else None,
                       "credit": os.environ.get("CAMERA_CREDIT") or None}}


@app.get("/api/live.mjpg")
def live_video():
    if live is None:
        raise HTTPException(404, "This camera source has no live view.")
    return StreamingResponse(live.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.post("/api/scan", status_code=202)
def scan_now():
    scanner.scan_now()
    return {"queued": True}


app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

"""The background loop: photo -> find the registered items -> save sightings."""
import logging
import os
import threading
import time

from .camera import CameraError

log = logging.getLogger("vlm_on_pi.scanner")


class Scanner(threading.Thread):
    def __init__(self, store, camera, detector, snapshot_dir, every_s=120, keep_days=7):
        super().__init__(name="scanner", daemon=True)
        self.store = store
        self.camera = camera
        self.detector = detector
        self.snapshot_dir = snapshot_dir
        self.every_s = every_s
        self.keep_days = keep_days
        self.busy = False
        self.last_started = None
        self._wake = threading.Event()
        # Not "_stop": threading.Thread already uses that name internally.
        self._stopping = threading.Event()
        os.makedirs(snapshot_dir, exist_ok=True)

    def scan_now(self):
        self._wake.set()

    def stop(self):
        self._stopping.set()
        self._wake.set()

    def run(self):
        while not self._stopping.is_set():
            started = self.last_started = time.time()
            try:
                self.scan_once()
            except Exception:  # keep the loop alive whatever one scan does
                log.exception("Scan failed")
            # A scan can take longer than the interval on a slow Pi; then the
            # next one starts straight away.
            self._wake.wait(max(0.0, self.every_s - (time.time() - started)))
            self._wake.clear()

    def scan_once(self):
        items = [i["name"] for i in self.store.items()]
        if not items:
            return None  # nothing to look for yet
        self.busy = True
        started = time.time()
        try:
            try:
                jpeg = self.camera.capture()
            except CameraError as e:
                return self.store.record_scan(started, None, None, [], error=str(e))
            name = "%d.jpg" % (started * 1000)
            self._write(name, jpeg)
            try:
                detections, info = self.detector.detect(jpeg, items)
            except OSError as e:
                return self.store.record_scan(started, round(time.time() - started, 1), name, [],
                                              error="The vision model didn't answer: %s" % e)
            log.info("Scan: %d of %d items found in %.1f s", len(detections), len(items), info["seconds"])
            # Yes/no for every item asked, so items added later read as "not checked yet".
            present = {item: item in info.get("present", []) for item in items}
            scan_id = self.store.record_scan(started, info["seconds"], name, detections, raw=info["raw"],
                                             present=present, stats=info.get("stats"))
            self._prune()
            return scan_id
        finally:
            self.busy = False

    def _write(self, name, jpeg):
        path = os.path.join(self.snapshot_dir, name)
        with open(path + ".tmp", "wb") as f:
            f.write(jpeg)
        os.replace(path + ".tmp", path)

    def _prune(self):
        for name in self.store.prune(time.time() - self.keep_days * 86400):
            try:
                os.remove(os.path.join(self.snapshot_dir, name))
            except OSError:
                pass

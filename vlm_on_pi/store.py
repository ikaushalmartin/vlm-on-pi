"""SQLite storage: items, zones, scans and sightings.

Everything is local to the device. Times are Unix seconds (UTC). Boxes and
zones are stored as JSON [x1, y1, x2, y2], normalized to the photo.
"""
import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    bbox TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY,
    started_at REAL NOT NULL,
    seconds REAL,
    snapshot TEXT,
    found INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    raw TEXT
);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY,
    scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    seen_at REAL NOT NULL,
    bbox TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sightings_item_time ON sightings(item_id, seen_at DESC);
"""


class Store:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        with self._connect() as db:
            db.executescript(SCHEMA)
            # Columns added after the first release; older databases get them here.
            columns = {r["name"] for r in db.execute("PRAGMA table_info(scans)")}
            for name in ("present", "stats"):
                if name not in columns:
                    db.execute("ALTER TABLE scans ADD COLUMN %s TEXT" % name)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        return db

    def _run(self, fn):
        # One writer at a time: the scanner thread and web requests share the file.
        with self._lock:
            db = self._connect()
            try:
                with db:
                    return fn(db)
            finally:
                db.close()

    # Items -----------------------------------------------------------------

    def add_item(self, name):
        name = " ".join(str(name).split())[:60]
        if not name:
            raise ValueError("Item name is empty.")

        def fn(db):
            db.execute("INSERT OR IGNORE INTO items (name, created_at) VALUES (?, ?)", (name, time.time()))
            return dict(db.execute("SELECT id, name FROM items WHERE name = ?", (name,)).fetchone())
        return self._run(fn)

    def remove_item(self, item_id):
        return self._run(lambda db: db.execute("DELETE FROM items WHERE id = ?", (item_id,)).rowcount > 0)

    def items(self):
        return self._run(lambda db: [dict(r) for r in db.execute("SELECT id, name FROM items ORDER BY name")])

    # Zones -----------------------------------------------------------------

    def add_zone(self, name, bbox):
        name = " ".join(str(name).split())[:60]
        if not name:
            raise ValueError("Zone name is empty.")

        def fn(db):
            cur = db.execute("INSERT INTO zones (name, bbox) VALUES (?, ?)", (name, json.dumps(bbox)))
            return {"id": cur.lastrowid, "name": name, "bbox": bbox}
        return self._run(fn)

    def remove_zone(self, zone_id):
        return self._run(lambda db: db.execute("DELETE FROM zones WHERE id = ?", (zone_id,)).rowcount > 0)

    def zones(self):
        rows = self._run(lambda db: db.execute("SELECT id, name, bbox FROM zones ORDER BY id").fetchall())
        return [{"id": r["id"], "name": r["name"], "bbox": json.loads(r["bbox"])} for r in rows]

    # Scans and sightings ---------------------------------------------------

    def record_scan(self, started_at, seconds, snapshot, detections, error=None, raw=None,
                    present=None, stats=None):
        """Save one scan and its sightings. `detections` hold item names;
        `present` maps each item asked about to whether the model said it
        was in view; `stats` is llama.cpp's token counts and timings per
        question."""
        def fn(db):
            ids = {r["name"].lower(): r["id"] for r in db.execute("SELECT id, name FROM items")}
            cur = db.execute(
                "INSERT INTO scans (started_at, seconds, snapshot, found, error, raw, present, stats)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (started_at, seconds, snapshot, len(detections), error, raw,
                 json.dumps(present) if present is not None else None,
                 json.dumps(stats) if stats is not None else None))
            scan_id = cur.lastrowid
            for d in detections:
                item_id = ids.get(d["item"].lower())
                if item_id is not None:  # the item may have been removed mid-scan
                    db.execute("INSERT INTO sightings (scan_id, item_id, seen_at, bbox) VALUES (?, ?, ?, ?)",
                               (scan_id, item_id, started_at, json.dumps(d["bbox"])))
            return scan_id
        return self._run(fn)

    def last_sighting(self, item_name):
        def fn(db):
            return db.execute(
                """SELECT i.name AS item, s.seen_at, s.bbox, c.snapshot
                   FROM sightings s JOIN items i ON i.id = s.item_id JOIN scans c ON c.id = s.scan_id
                   WHERE i.name = ? ORDER BY s.seen_at DESC LIMIT 1""", (item_name,)).fetchone()
        row = self._run(fn)
        return _sighting(row) if row else None

    def last_sightings(self):
        """The latest sighting of every item (items never seen are left out)."""
        def fn(db):
            return db.execute(
                """SELECT i.name AS item, s.seen_at, s.bbox, c.snapshot
                   FROM items i
                   JOIN sightings s ON s.id = (SELECT id FROM sightings WHERE item_id = i.id
                                               ORDER BY seen_at DESC LIMIT 1)
                   JOIN scans c ON c.id = s.scan_id
                   ORDER BY i.name""").fetchall()
        return [_sighting(r) for r in self._run(fn)]

    def last_scan(self):
        def fn(db):
            return db.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
        row = self._run(fn)
        if not row:
            return None
        scan = dict(row)
        scan["present"] = json.loads(scan["present"]) if scan.get("present") else None
        scan["stats"] = json.loads(scan["stats"]) if scan.get("stats") else []
        scan["detections"] = [
            {"item": r["name"], "bbox": json.loads(r["bbox"])}
            for r in self._run(lambda db: db.execute(
                "SELECT i.name, s.bbox FROM sightings s JOIN items i ON i.id = s.item_id WHERE s.scan_id = ?",
                (scan["id"],)).fetchall())]
        return scan

    def prune(self, older_than):
        """Drop scans (and their sightings) older than `older_than`, except
        each item's latest sighting, so "last seen" survives a quiet week.
        Returns the snapshot file names that are no longer referenced."""
        def fn(db):
            keep = """SELECT scan_id FROM sightings WHERE id IN (
                          SELECT (SELECT id FROM sightings WHERE item_id = i.id ORDER BY seen_at DESC LIMIT 1)
                          FROM items i)"""
            old = db.execute(
                "SELECT id, snapshot FROM scans WHERE started_at < ? AND id NOT IN (%s)" % keep,
                (older_than,)).fetchall()
            db.executemany("DELETE FROM scans WHERE id = ?", [(r["id"],) for r in old])
            still_used = {r["snapshot"] for r in db.execute("SELECT snapshot FROM scans WHERE snapshot IS NOT NULL")}
            return sorted({r["snapshot"] for r in old if r["snapshot"]} - still_used)
        return self._run(fn)


def _sighting(row):
    return {"item": row["item"], "seen_at": row["seen_at"], "bbox": json.loads(row["bbox"]),
            "snapshot": row["snapshot"]}

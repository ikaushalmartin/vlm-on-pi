"""Tests for the parts that don't need a camera or the model.

Run with:  python3 -m unittest discover -s tests
"""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vlm_on_pi.ask import item_in_question  # noqa: E402
from vlm_on_pi.camera import CameraError  # noqa: E402
from vlm_on_pi.detector import Detector, clean_box, match_item, parse_detections, parse_presence  # noqa: E402
from vlm_on_pi.places import ago, describe, place_of, rough_position  # noqa: E402
from vlm_on_pi.scanner import Scanner  # noqa: E402
from vlm_on_pi.store import Store  # noqa: E402
from vlm_on_pi.stream import split_jpegs  # noqa: E402


class SplitJpegs(unittest.TestCase):
    A = b"\xff\xd8aaa\xff\xd9"
    B = b"\xff\xd8bbbb\xff\xd9"

    def test_whole_frames(self):
        self.assertEqual(split_jpegs(self.A + self.B), ([self.A, self.B], b""))

    def test_frame_split_across_reads(self):
        jpegs, rest = split_jpegs(self.A + self.B[:4])
        self.assertEqual(jpegs, [self.A])
        jpegs, rest = split_jpegs(rest + self.B[4:])
        self.assertEqual((jpegs, rest), ([self.B], b""))

    def test_start_marker_split_across_reads(self):
        jpegs, rest = split_jpegs(b"junk\xff")
        self.assertEqual((jpegs, rest), ([], b"\xff"))
        self.assertEqual(split_jpegs(rest + self.A[1:]), ([self.A], b""))


class ParseDetections(unittest.TestCase):
    ITEMS = ["cat", "remote control", "car keys"]

    def test_clean_answer(self):
        text = '[{"label": "cat", "bbox": [0.0, 0.16, 0.5, 0.95]}, {"label": "remote control", "bbox": [0.2, 0.17, 0.4, 0.33]}]'
        found = parse_detections(text, self.ITEMS)
        self.assertEqual([d["item"] for d in found], ["cat", "remote control"])
        self.assertEqual(found[0]["bbox"], [0.0, 0.16, 0.5, 0.95])

    def test_answer_cut_off_at_the_token_limit(self):
        # Seen on the Pi: the closing "]" was missing.
        text = '[{"label": "cat", "bbox": [0.0, 0.18, 0.49, 0.98]}, {"label": "remote control", "bbox": [0.12, 0.17, 0.36, 0.3]}'
        self.assertEqual(len(parse_detections(text, self.ITEMS)), 2)

    def test_half_written_object_is_dropped(self):
        text = '[{"label": "cat", "bbox": [0.1, 0.1, 0.5, 0.5]}, {"label": "remote control", "bbox": [0.2, 0.'
        self.assertEqual([d["item"] for d in parse_detections(text, self.ITEMS)], ["cat"])

    def test_text_around_the_json(self):
        text = 'Sure! Here you go:\n```json\n[{"label": "Car Keys", "bbox": [0.5, 0.5, 0.6, 0.6]}]\n```'
        self.assertEqual(parse_detections(text, self.ITEMS)[0]["item"], "car keys")

    def test_unregistered_labels_and_bad_boxes_are_dropped(self):
        text = ('[{"label": "dog", "bbox": [0.1, 0.1, 0.2, 0.2]},'
                ' {"label": "cat", "bbox": [0.5, 0.5, 0.4, 0.6]},'
                ' {"label": "cat", "bbox": [0.1, 0.1]},'
                ' {"label": "cat", "bbox": "nowhere"}]')
        self.assertEqual(parse_detections(text, self.ITEMS), [])

    def test_no_json_at_all(self):
        self.assertEqual(parse_detections("I can't see anything.", self.ITEMS), [])
        self.assertEqual(parse_detections("", self.ITEMS), [])
        self.assertEqual(parse_detections(None, self.ITEMS), [])

    def test_boxes_are_clamped(self):
        self.assertEqual(clean_box([-0.01, 0.2, 1.02, 0.9]), [0.0, 0.2, 1.0, 0.9])

    def test_label_matching(self):
        self.assertEqual(match_item("remote", ["remote control"]), "remote control")
        self.assertEqual(match_item("keys", ["key"]), "key")
        self.assertEqual(match_item("car keys", ["keys", "car keys"]), "car keys")
        self.assertIsNone(match_item("phone", ["keys"]))
        self.assertIsNone(match_item("", ["keys"]))


class Presence(unittest.TestCase):
    ITEMS = ["tv", "cat", "remote control", "clock"]

    def test_real_answer_from_the_pi(self):
        # Seen on the Pi, cut at the closing brace.
        text = ('```json { "tv": true, "cat": false, "remote control": true, "clock": true, '
                '"potted plant": true, "person": true }')
        self.assertEqual(parse_presence(text, self.ITEMS), ["tv", "remote control", "clock"])

    def test_names_are_matched_loosely_and_kept_in_item_order(self):
        self.assertEqual(parse_presence('{"Clocks": true, "TV": TRUE}', self.ITEMS), ["tv", "clock"])

    def test_nothing_usable(self):
        self.assertEqual(parse_presence("Yes", self.ITEMS), [])
        self.assertEqual(parse_presence(None, self.ITEMS), [])


class Imaging(unittest.TestCase):
    def test_jpeg_size_from_header(self):
        import struct
        from vlm_on_pi.imaging import jpeg_size
        app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, 286, 512, 1) + b"\x01\x11\x00"
        self.assertEqual(jpeg_size(b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"), (512, 286))
        with self.assertRaises(ValueError):
            jpeg_size(b"not a jpeg")

    def test_zoom_region_is_twice_the_box_and_stays_in_frame(self):
        from vlm_on_pi.imaging import zoom_region
        left, top, w, h = zoom_region([0.4, 0.4, 0.6, 0.5])
        self.assertAlmostEqual(w, 0.4)
        self.assertAlmostEqual(h, 0.25)  # 0.2 would be under the 25% minimum
        self.assertAlmostEqual(left, 0.3)
        self.assertEqual(zoom_region([0.0, 0.9, 0.1, 1.0])[:2], (0.0, 0.75))  # pushed back inside

    def test_unzoom_maps_back_to_the_frame(self):
        from vlm_on_pi.imaging import unzoom
        self.assertEqual(unzoom([0.5, 0.5, 1.0, 1.0], (0.2, 0.4, 0.4, 0.2)), [0.4, 0.5, 0.6, 0.6])


class ZoomIn(unittest.TestCase):
    def detector(self, answers):
        d = Detector("http://unused", crop=lambda jpeg, region: b"crop")
        replies = iter(answers)
        d._ask = lambda image_url, prompt, stop=None: (next(replies), {})
        return d

    def test_box_is_refined_from_the_crop(self):
        d = self.detector(['{"tv": true', '[{"label": "tv", "bbox": [0.4, 0.4, 0.6, 0.5]}]',
                           '[{"label": "tv", "bbox": [0.25, 0.0, 0.75, 1.0]}]'])
        found, info = d.detect(b"jpeg", ["tv"])
        # Crop region is (0.3, 0.325, 0.4, 0.25); the refined box maps back into it.
        self.assertEqual(found[0]["bbox"], [0.4, 0.325, 0.6, 0.575])
        self.assertEqual([q["question"] for q in info["stats"]], ["present", "boxes", "zoom"])

    def test_rough_box_kept_when_the_crop_finds_nothing(self):
        d = self.detector(['{"tv": true', '[{"label": "tv", "bbox": [0.4, 0.4, 0.6, 0.5]}]', "[]"])
        found, _ = d.detect(b"jpeg", ["tv"])
        self.assertEqual(found[0]["bbox"], [0.4, 0.4, 0.6, 0.5])


class TwoQuestionDetect(unittest.TestCase):
    def detector(self, presence, boxes):
        d = Detector("http://unused", zoom=False)
        d.asked = []

        def fake_ask(image_url, prompt, stop=None):
            d.asked.append(prompt)
            first = len(d.asked) == 1
            # Shaped like llama.cpp's "timings" from the Pi.
            timings = {"prompt_n": 207 if first else 219, "prompt_ms": 12374.3 if first else 12715.8,
                       "predicted_n": 85 if first else 135, "predicted_ms": 6627.3 if first else 10728.7}
            return (presence if first else boxes), timings
        d._ask = fake_ask
        return d

    def test_boxes_only_for_present_items(self):
        d = self.detector(
            '{"tv": true, "cat": false, "clock": true',
            # The model draws a box for the cat anyway; it must be dropped.
            '[{"label": "tv", "bbox": [0.04, 0.38, 0.24, 0.66]}, {"label": "cat", "bbox": [0.6, 0.36, 0.66, 0.62]}]')
        found, info = d.detect(b"jpeg", ["tv", "cat", "clock"])
        self.assertEqual([f["item"] for f in found], ["tv"])
        self.assertEqual(info["present"], ["tv", "clock"])
        self.assertIn("Detect all instances of: tv, clock.", d.asked[1])

    def test_token_stats_per_question(self):
        d = self.detector('{"tv": true', '[{"label": "tv", "bbox": [0.1, 0.1, 0.2, 0.2]}]')
        _, info = d.detect(b"jpeg", ["tv"])
        self.assertEqual(info["stats"], [
            {"question": "present", "prompt_tokens": 207, "prompt_s": 12.4, "answer_tokens": 85, "answer_s": 6.6},
            {"question": "boxes", "prompt_tokens": 219, "prompt_s": 12.7, "answer_tokens": 135, "answer_s": 10.7}])

    def test_nothing_present_skips_the_box_question(self):
        d = self.detector('{"tv": false, "cat": false', "unused")
        found, info = d.detect(b"jpeg", ["tv", "cat"])
        self.assertEqual((found, info["present"], len(d.asked)), ([], [], 1))


class Ask(unittest.TestCase):
    def test_finds_the_item(self):
        self.assertEqual(item_in_question("Where are my keys?", ["keys", "wallet"]), "keys")
        self.assertEqual(item_in_question("wheres my WALLET", ["keys", "wallet"]), "wallet")

    def test_longest_match_wins(self):
        self.assertEqual(item_in_question("where are the car keys", ["keys", "car keys"]), "car keys")

    def test_plurals(self):
        self.assertEqual(item_in_question("where are my glasses?", ["glasses"]), "glasses")
        self.assertEqual(item_in_question("where are my glasses?", ["glass"]), "glass")
        self.assertEqual(item_in_question("where is my key", ["keys"]), "keys")
        self.assertEqual(item_in_question("where are the batteries", ["battery"]), "battery")
        self.assertEqual(item_in_question("where's my watch", ["watches"]), "watches")

    def test_whole_words_only(self):
        self.assertIsNone(item_in_question("where is the monkey?", ["key"]))

    def test_unknown(self):
        self.assertIsNone(item_in_question("where is my phone?", ["keys"]))


class Places(unittest.TestCase):
    ZONES = [{"name": "on the desk", "bbox": [0.0, 0.5, 1.0, 1.0]},
             {"name": "in the drawer", "bbox": [0.1, 0.6, 0.3, 0.8]}]

    def test_zone(self):
        self.assertEqual(place_of([0.6, 0.6, 0.7, 0.7], self.ZONES), "on the desk")

    def test_smallest_zone_wins(self):
        self.assertEqual(place_of([0.15, 0.65, 0.25, 0.75], self.ZONES), "in the drawer")

    def test_no_zone_gives_a_rough_position(self):
        self.assertEqual(place_of([0.0, 0.0, 0.1, 0.1], self.ZONES), "at the top left of the view")
        self.assertEqual(rough_position([0.4, 0.4, 0.6, 0.6]), "in the middle of the view")
        self.assertEqual(rough_position([0.8, 0.4, 0.9, 0.6]), "on the right side of the view")
        self.assertEqual(rough_position([0.4, 0.9, 0.6, 1.0]), "at the bottom middle of the view")

    def test_ago(self):
        self.assertEqual(ago(5), "just now")
        self.assertEqual(ago(12 * 60), "12 min ago")
        self.assertEqual(ago(3 * 3600), "3 h ago")
        self.assertEqual(ago(30 * 3600), "yesterday")
        self.assertEqual(ago(3 * 86400), "3 days ago")

    def test_describe(self):
        s = {"item": "keys", "bbox": [0.6, 0.6, 0.7, 0.7], "seen_at": 1000.0}
        self.assertEqual(describe(s, self.ZONES, now=1000.0 + 12 * 60), "keys — last seen on the desk, 12 min ago.")
        s.update(item="tv", bbox=[0.4, 0.0, 0.6, 0.2])
        self.assertEqual(describe(s, [], now=1000.0), "tv — last seen at the top middle of the view, just now.")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.dir.name, "t.db"))

    def tearDown(self):
        self.dir.cleanup()

    def test_presence_and_stats_are_kept_per_look(self):
        self.store.add_item("tv")
        stats = [{"question": "present", "prompt_tokens": 207, "prompt_s": 12.4, "answer_tokens": 85, "answer_s": 6.6}]
        self.store.record_scan(100.0, 19.0, "a.jpg", [], present={"tv": False}, stats=stats)
        scan = self.store.last_scan()
        self.assertEqual((scan["present"], scan["stats"]), ({"tv": False}, stats))

    def test_old_database_gets_the_new_columns(self):
        import sqlite3
        path = os.path.join(self.dir.name, "old.db")
        db = sqlite3.connect(path)
        db.executescript("CREATE TABLE scans (id INTEGER PRIMARY KEY, started_at REAL NOT NULL, seconds REAL,"
                         " snapshot TEXT, found INTEGER NOT NULL DEFAULT 0, error TEXT, raw TEXT);"
                         "INSERT INTO scans (started_at, seconds, snapshot) VALUES (1.0, 40.0, 'old.jpg');")
        db.commit()
        db.close()
        old = Store(path)
        self.assertEqual((old.last_scan()["present"], old.last_scan()["stats"]), (None, []))
        old.record_scan(2.0, 41.0, "new.jpg", [], present={}, stats=[])
        self.assertEqual(old.last_scan()["snapshot"], "new.jpg")

    def test_items_are_unique_ignoring_case(self):
        a = self.store.add_item("Keys")
        b = self.store.add_item("keys")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(len(self.store.items()), 1)
        with self.assertRaises(ValueError):
            self.store.add_item("   ")

    def test_last_sighting_is_the_newest(self):
        self.store.add_item("keys")
        self.store.record_scan(100.0, 40.0, "a.jpg", [{"item": "keys", "bbox": [0.1, 0.1, 0.2, 0.2]}])
        self.store.record_scan(200.0, 40.0, "b.jpg", [{"item": "keys", "bbox": [0.5, 0.5, 0.6, 0.6]}])
        self.store.record_scan(300.0, 40.0, "c.jpg", [])
        s = self.store.last_sighting("keys")
        self.assertEqual((s["seen_at"], s["snapshot"], s["bbox"]), (200.0, "b.jpg", [0.5, 0.5, 0.6, 0.6]))
        self.assertIsNone(self.store.last_sighting("wallet"))

    def test_removing_an_item_removes_its_sightings(self):
        item = self.store.add_item("keys")
        self.store.record_scan(100.0, 40.0, "a.jpg", [{"item": "keys", "bbox": [0.1, 0.1, 0.2, 0.2]}])
        self.store.remove_item(item["id"])
        self.assertEqual(self.store.last_sightings(), [])

    def test_sighting_for_an_item_removed_mid_scan_is_skipped(self):
        self.store.record_scan(100.0, 40.0, "a.jpg", [{"item": "ghost", "bbox": [0.1, 0.1, 0.2, 0.2]}])
        self.assertEqual(self.store.last_scan()["detections"], [])

    def test_prune_keeps_each_items_latest_sighting(self):
        self.store.add_item("keys")
        self.store.add_item("wallet")
        self.store.record_scan(100.0, 1, "old-keys.jpg", [{"item": "keys", "bbox": [0.1, 0.1, 0.2, 0.2]}])
        self.store.record_scan(150.0, 1, "old-wallet.jpg", [{"item": "wallet", "bbox": [0.1, 0.1, 0.2, 0.2]}])
        self.store.record_scan(200.0, 1, "empty.jpg", [])
        self.store.record_scan(900.0, 1, "new-keys.jpg", [{"item": "keys", "bbox": [0.3, 0.3, 0.4, 0.4]}])
        removed = self.store.prune(older_than=500.0)
        self.assertEqual(removed, ["empty.jpg", "old-keys.jpg"])
        self.assertEqual(self.store.last_sighting("wallet")["snapshot"], "old-wallet.jpg")
        self.assertEqual(self.store.last_sighting("keys")["snapshot"], "new-keys.jpg")

    def test_zones(self):
        z = self.store.add_zone("on the desk", [0.0, 0.5, 1.0, 1.0])
        self.assertEqual(self.store.zones(), [z])
        self.assertTrue(self.store.remove_zone(z["id"]))
        self.assertFalse(self.store.remove_zone(z["id"]))


class FakeCamera:
    def __init__(self, error=None):
        self.error = error

    def capture(self):
        if self.error:
            raise CameraError(self.error)
        return b"\xff\xd8fake-jpeg"


class FakeDetector:
    def __init__(self, found=None, error=None):
        self.found = found or []
        self.error = error
        self.asked = None

    def detect(self, jpeg, items):
        self.asked = items
        if self.error:
            raise self.error
        found = [d for d in self.found if d["item"] in items]
        return found, {"seconds": 41.0, "raw": "[...]", "present": [d["item"] for d in found], "stats": []}


class ScannerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = Store(os.path.join(self.dir.name, "t.db"))
        self.snaps = os.path.join(self.dir.name, "snaps")

    def tearDown(self):
        self.dir.cleanup()

    def scanner(self, camera=None, detector=None):
        return Scanner(self.store, camera or FakeCamera(), detector or FakeDetector(), self.snaps)

    def test_nothing_to_look_for(self):
        self.assertIsNone(self.scanner().scan_once())
        self.assertIsNone(self.store.last_scan())

    def test_a_scan_saves_the_photo_and_sightings(self):
        self.store.add_item("keys")
        self.store.add_item("wallet")
        detector = FakeDetector([{"item": "keys", "bbox": [0.1, 0.1, 0.2, 0.2]}])
        self.scanner(detector=detector).scan_once()
        scan = self.store.last_scan()
        self.assertEqual(detector.asked, ["keys", "wallet"])
        self.assertEqual((scan["found"], scan["seconds"], scan["error"]), (1, 41.0, None))
        self.assertEqual(scan["present"], {"keys": True, "wallet": False})
        self.assertTrue(os.path.exists(os.path.join(self.snaps, scan["snapshot"])))
        self.assertEqual(self.store.last_sighting("keys")["bbox"], [0.1, 0.1, 0.2, 0.2])

    def test_camera_failure_is_recorded(self):
        self.store.add_item("keys")
        self.scanner(camera=FakeCamera("unplugged")).scan_once()
        self.assertEqual(self.store.last_scan()["error"], "unplugged")

    def test_model_failure_is_recorded(self):
        self.store.add_item("keys")
        self.scanner(detector=FakeDetector(error=ConnectionRefusedError("refused"))).scan_once()
        self.assertIn("didn't answer", self.store.last_scan()["error"])

    def test_loop_runs_and_stops(self):
        self.store.add_item("keys")
        scanner = self.scanner()
        scanner.every_s = 3600
        scanner.start()
        deadline = time.time() + 5
        while self.store.last_scan() is None and time.time() < deadline:
            time.sleep(0.05)
        scanner.stop()
        scanner.join(timeout=5)
        self.assertFalse(scanner.is_alive())
        self.assertIsNotNone(self.store.last_scan())


if __name__ == "__main__":
    unittest.main()

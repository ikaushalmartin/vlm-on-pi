"""Ask the vision model where the registered items are in one photo.

Each look is two questions about the same photo:
  1. Which of the items are in it? (a JSON map of item -> true/false)
  2. Where are the ones that are? (a JSON array of {"label", "bbox"},
     boxes normalized to [0, 1])

Asked only to draw boxes, this small model (LFM2.5-VL-450M) "finds" every item
it's asked for, even ones that aren't there: on a test photo it put a cat and a
remote control on a person. Asked whether each item is present, it's right far
more often, so boxes are only requested for items it says are present.

Its boxes on a wide CCTV frame are rough (the frame is only ~144 image tokens),
so each box then gets a zoom-in pass: crop around it and ask again at full
detail. On a test frame that doubled the overlap with the true positions.

Small models also add text around the JSON, get cut off at the token limit or
name an item slightly differently, so parsing is defensive.
"""
import base64
import json
import re
import time
import urllib.request

from .imaging import crop_jpeg, unzoom, zoom_region

PRESENCE_PROMPT = ("Which of these are in the image: {items}? "
                   "Answer with a JSON object that maps each one to true or false.")

# The grounding prompt format this model expects.
PROMPT = ('Detect all instances of: {items}. Response must be a JSON array: '
          '[{{"label": ..., "bbox": [x1, y1, x2, y2]}}, ...]. '
          'Coordinates are normalized to [0,1].')

_OBJECT = re.compile(r"\{[^{}]*\}")
_PRESENCE_PAIR = re.compile(r'"([^"]+)"\s*:\s*(true|false)', re.I)


def _singular(word):
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word  # "glass", "bus", "iris"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"  # "batteries" -> "battery"
    if word.endswith(("sses", "ches", "shes", "xes", "zes")):
        return word[:-2]  # "glasses" -> "glass", "watches" -> "watch"
    if word.endswith("s"):
        return word[:-1]  # "keys" -> "key"
    return word


def normalize(name):
    """Lower-case, single-spaced and singular ("Car Keys" -> "car key")."""
    words = re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).split()
    return " ".join(_singular(w) for w in words)


def match_item(label, items):
    """The registered item a model label refers to, or None.

    An exact match wins; otherwise the longest item whose name contains, or is
    contained in, the label ("remote" matches the label "remote control").
    """
    wanted = normalize(label)
    if not wanted:
        return None
    best = None
    for item in items:
        name = normalize(item)
        if name == wanted:
            return item
        if name and (name in wanted or wanted in name):
            if best is None or len(name) > len(normalize(best)):
                best = item
    return best


def clean_box(box):
    """[x1, y1, x2, y2] clamped to [0, 1], or None if it isn't a real box."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def parse_presence(text, items):
    """Model output -> the registered items it says are present, in `items` order."""
    present = set()
    for key, value in _PRESENCE_PAIR.findall(text or ""):
        item = match_item(key, items)
        if item is not None and value.lower() == "true":
            present.add(item)
    return [item for item in items if item in present]


def parse_detections(text, items):
    """Model output -> [{"item", "label", "bbox"}] for registered items only."""
    found = []
    for chunk in _OBJECT.findall(text or ""):
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue
        item = match_item(obj.get("label", ""), items)
        box = clean_box(obj.get("bbox"))
        if item is not None and box is not None:
            found.append({"item": item, "label": str(obj["label"]), "bbox": box})
    return found


def _stats(question, timings):
    """llama.cpp's per-request timings, reduced to what the page shows."""
    return {
        "question": question,
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_s": round(timings.get("prompt_ms", 0) / 1000.0, 1),
        "answer_tokens": timings.get("predicted_n"),
        "answer_s": round(timings.get("predicted_ms", 0) / 1000.0, 1),
    }


def _data_url(jpeg):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


class Detector:
    def __init__(self, llm_url, max_tokens=256, timeout_s=600, zoom=True, crop=crop_jpeg):
        self.llm_url = llm_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.zoom = zoom
        self._crop = crop

    def detect(self, jpeg, items):
        """Find `items` in one JPEG. Returns (detections, info) where info has
        the raw model text, which items it said were present, how long the
        look took, and llama.cpp's token counts and timings per question.

        Each question re-reads the photo: llama.cpp can't reuse a cached
        photo for this model's hybrid layers. So it is two questions per
        look, not one per item.
        """
        image_url = _data_url(jpeg)
        started = time.time()
        # Stop at the end of the JSON map; the model likes to explain itself after it.
        answer, timings = self._ask(image_url, PRESENCE_PROMPT.format(items=", ".join(items)), stop=["}"])
        answer += "}"
        present = parse_presence(answer, items)
        raw = ["present: " + " ".join(answer.split())]
        stats = [_stats("present", timings)]
        found = []
        if present:
            boxes, timings = self._ask(image_url, PROMPT.format(items=", ".join(present)))
            raw.append("boxes: " + " ".join(boxes.split()))
            stats.append(_stats("boxes", timings))
            found = parse_detections(boxes, present)
            if self.zoom:
                found = [self._zoom_in(jpeg, det, raw, stats) for det in found]
        info = {"seconds": round(time.time() - started, 1), "raw": "\n".join(raw),
                "present": present, "stats": stats}
        return found, info

    def _zoom_in(self, jpeg, det, raw, stats):
        """Re-ask for one item on a crop around its rough box. Keeps the rough
        box if the crop fails or the model finds nothing in it."""
        region = zoom_region(det["bbox"])
        try:
            crop = self._crop(jpeg, region)
        except (OSError, ValueError):
            return det
        text, timings = self._ask(_data_url(crop), PROMPT.format(items=det["item"]))
        raw.append("zoom %s: %s" % (det["item"], " ".join(text.split())))
        stats.append(_stats("zoom", timings))
        again = parse_detections(text, [det["item"]])
        if not again:
            return det
        return dict(det, bbox=clean_box(unzoom(again[0]["bbox"], region)) or det["bbox"])

    def _ask(self, image_url, prompt, stop=None):
        """One question about the photo -> (answer text, llama.cpp timings)."""
        body = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]}],
            # Recommended sampling settings for this model.
            "temperature": 0.1,
            "min_p": 0.15,
            "repeat_penalty": 1.05,
            "max_tokens": self.max_tokens,
            # Every look is a new photo, so there's nothing useful to cache.
            "cache_prompt": False,
        }
        if stop:
            body["stop"] = stop
        request = urllib.request.Request(
            self.llm_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"], result.get("timings") or {}

    def healthy(self):
        try:
            with urllib.request.urlopen(self.llm_url + "/health", timeout=3) as response:
                return response.status == 200
        except OSError:
            return False

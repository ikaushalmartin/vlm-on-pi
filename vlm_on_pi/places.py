"""Turn a box into words: a named zone ("on the shelf") or a rough position."""
import time


def center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def zone_for(bbox, zones):
    """The most specific zone containing the box's center, or None.

    `zones` are dicts with "name" and "bbox". When zones overlap (a "drawer"
    inside the "desk"), the smallest one wins.
    """
    cx, cy = center(bbox)
    containing = [z for z in zones
                  if z["bbox"][0] <= cx <= z["bbox"][2] and z["bbox"][1] <= cy <= z["bbox"][3]]
    if not containing:
        return None
    area = lambda z: (z["bbox"][2] - z["bbox"][0]) * (z["bbox"][3] - z["bbox"][1])
    return min(containing, key=area)


def rough_position(bbox):
    """"at the top left of the view", "in the middle of the view"... from a 3x3 grid."""
    cx, cy = center(bbox)
    row = ("top", "middle", "bottom")[min(2, int(cy * 3))]
    col = ("left", "middle", "right")[min(2, int(cx * 3))]
    if row == col == "middle":
        return "in the middle of the view"
    if col == "middle":
        return "at the %s middle of the view" % row
    if row == "middle":
        return "on the %s side of the view" % col
    return "at the %s %s of the view" % (row, col)


def place_of(bbox, zones):
    # Zones are named the way you'd say them ("on the desk", "in the drawer"),
    # so the name is used as-is.
    zone = zone_for(bbox, zones)
    return zone["name"] if zone else rough_position(bbox)


def ago(seconds):
    """A short "how long ago" in plain words."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return "%d min ago" % minutes
    hours = minutes // 60
    if hours < 24:
        return "%d h ago" % hours
    days = hours // 24
    return "yesterday" if days == 1 else "%d days ago" % days


def describe(sighting, zones, now=None):
    """'keys — last seen on the shelf, 12 min ago.' The item keeps the user's
    own spelling ("tv", "iPad"), and the dash avoids is/are grammar."""
    now = time.time() if now is None else now
    return "%s — last seen %s, %s." % (
        sighting["item"], place_of(sighting["bbox"], zones), ago(now - sighting["seen_at"]))

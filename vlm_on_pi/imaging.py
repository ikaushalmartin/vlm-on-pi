"""Small JPEG helpers for the zoom-in pass: read a JPEG's size, crop a region."""
import struct
import subprocess

# Start-of-frame markers that carry the image size (baseline, progressive, ...).
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(data):
    """(width, height) from a JPEG's header, without decoding it."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in _SOF:
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    raise ValueError("Not a JPEG, or no size in its header.")


def zoom_region(box, factor=2.0, minimum=0.25):
    """The region to crop around a rough box: `factor` times its size (at
    least `minimum` of the frame each way), centred on it and kept inside the
    frame. Returns (left, top, width, height), all as fractions of the frame."""
    x1, y1, x2, y2 = box
    width = min(1.0, max(minimum, factor * (x2 - x1)))
    height = min(1.0, max(minimum, factor * (y2 - y1)))
    left = min(max(0.0, (x1 + x2) / 2 - width / 2), 1.0 - width)
    top = min(max(0.0, (y1 + y2) / 2 - height / 2), 1.0 - height)
    return left, top, width, height


def unzoom(box, region):
    """A box found inside the crop, mapped back to the whole frame."""
    left, top, width, height = region
    return [round(left + box[0] * width, 4), round(top + box[1] * height, 4),
            round(left + box[2] * width, 4), round(top + box[3] * height, 4)]


def crop_jpeg(data, region, width=384):
    """Crop `region` out of a JPEG and scale it to `width` pixels wide."""
    w, h = jpeg_size(data)
    left, top, cw, ch = region
    vf = "crop=%d:%d:%d:%d,scale=%d:-2" % (max(2, cw * w), max(2, ch * h), left * w, top * h, width)
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-i", "pipe:0", "-vf", vf,
         "-frames:v", "1", "-f", "image2", "-vcodec", "mjpeg", "-q:v", "3", "pipe:1"],
        input=data, capture_output=True, timeout=15)
    if result.returncode != 0 or not result.stdout:
        raise ValueError("Couldn't crop the frame.")
    return result.stdout

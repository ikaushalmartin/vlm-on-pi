"""Grab one photo as a JPEG, sized for the model.

The source can be:
  /dev/video0            a USB webcam (or the Pi camera through V4L2)
  rtsp://... or http://  a network camera
  folder:/path           cycle through the images in a folder (demo and testing)

ffmpeg does the capture and resizing, so there is no OpenCV dependency.
"""
import os
import subprocess

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, source, size=512, warmup_frames=15):
        self.source = source
        self.size = size
        # Webcams start dark while auto-exposure settles; skip the first frames.
        self.warmup_frames = warmup_frames
        self._next = 0

    def capture(self):
        if self.source.startswith("folder:"):
            return self._from_folder(self.source[len("folder:"):])
        if self.source.startswith(("rtsp://", "rtsps://")):
            args = ["-rtsp_transport", "tcp", "-i", self.source]
            vf = self._scale()
        elif self.source.startswith(("http://", "https://")):
            args = ["-i", self.source]
            vf = self._scale()
        else:
            args = ["-f", "v4l2", "-i", self.source]
            vf = "select=gte(n\\,%d),%s" % (self.warmup_frames, self._scale())
        return self._ffmpeg(args + ["-vf", vf])

    def _scale(self):
        # Longest side = self.size, aspect ratio kept.
        return "scale=w=%d:h=%d:force_original_aspect_ratio=decrease" % (self.size, self.size)

    def _ffmpeg(self, args, timeout=30):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args + [
            "-frames:v", "1", "-f", "image2", "-vcodec", "mjpeg", "-q:v", "3", "pipe:1"]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise CameraError("The camera didn't send a picture within %d s." % timeout)
        except FileNotFoundError:
            raise CameraError("ffmpeg isn't installed.")
        if result.returncode != 0 or not result.stdout:
            message = result.stderr.decode(errors="replace").strip().splitlines()
            raise CameraError("Couldn't read the camera %s: %s" % (
                self.source, message[-1] if message else "no picture"))
        return result.stdout

    def _from_folder(self, folder):
        try:
            names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTENSIONS))
        except OSError as e:
            raise CameraError("Can't open the demo folder %s: %s" % (folder, e))
        if not names:
            raise CameraError("The demo folder %s has no images." % folder)
        path = os.path.join(folder, names[self._next % len(names)])
        self._next += 1
        return self._ffmpeg(["-i", path, "-vf", self._scale()])

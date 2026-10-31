"""A live camera stream shared by the web page and the looks.

One ffmpeg process reads the camera and turns it into a steady stream of small
JPEGs. The page shows them as live video (MJPEG); a look takes the newest one.
Only one reader touches the camera, so a USB webcam is never "busy".

The source can be a USB webcam (/dev/video0), a network camera (rtsp://,
http://) or a video file, which is played in real time on a loop, like a
recording standing in for a camera.
"""
import logging
import subprocess
import threading
import time

from .camera import CameraError

log = logging.getLogger("vlm_on_pi.stream")

SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def split_jpegs(buffer):
    """Cut complete JPEGs out of a byte stream. Returns (jpegs, leftover bytes)."""
    jpegs = []
    while True:
        start = buffer.find(SOI)
        if start < 0:
            # Keep a trailing 0xFF: it may be the first half of the next SOI.
            return jpegs, buffer[-1:] if buffer.endswith(b"\xff") else b""
        end = buffer.find(EOI, start + 2)
        if end < 0:
            return jpegs, buffer[start:]
        jpegs.append(buffer[start:end + 2])
        buffer = buffer[end + 2:]


class LiveSource:
    def __init__(self, source, width=640, fps=10, look_size=512):
        self.source = source
        self.width = width
        self.fps = fps
        self.look_size = look_size
        self.error = None
        self._frame = None
        self._count = 0
        self._at = 0.0
        self._cond = threading.Condition()
        self._stopping = threading.Event()
        self._proc = None
        self._thread = threading.Thread(target=self._run, name="live", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stopping.set()
        if self._proc:
            self._proc.kill()

    @property
    def live(self):
        return self._frame is not None and time.time() - self._at < 3

    def _input(self):
        s = self.source
        if s.startswith(("rtsp://", "rtsps://")):
            return ["-rtsp_transport", "tcp", "-i", s]
        if s.startswith(("http://", "https://")):
            return ["-i", s]
        if s.startswith("/dev/"):
            return ["-f", "v4l2", "-i", s]
        # A video file: play it at its real speed, forever.
        return ["-re", "-stream_loop", "-1", "-i", s]

    def _run(self):
        cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error"] + self._input() +
               ["-an", "-vf", "fps=%d,scale=%d:-2" % (self.fps, self.width),
                "-q:v", "5", "-f", "mjpeg", "pipe:1"])
        while not self._stopping.is_set():
            try:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                self.error = "ffmpeg isn't installed."
                return
            leftover = b""
            while not self._stopping.is_set():
                chunk = self._proc.stdout.read(65536)
                if not chunk:
                    break
                jpegs, leftover = split_jpegs(leftover + chunk)
                for jpeg in jpegs:
                    self._publish(jpeg)
            self._proc.kill()
            message = self._proc.stderr.read().decode(errors="replace").strip().splitlines()
            if not self._stopping.is_set():
                self.error = "Camera stream stopped%s. Retrying." % (": " + message[-1] if message else "")
                log.warning(self.error)
                self._stopping.wait(3)

    def _publish(self, jpeg):
        with self._cond:
            self._frame = jpeg
            self._count += 1
            self._at = time.time()
            self.error = None
            self._cond.notify_all()

    def next_frame(self, after, timeout=5):
        """Wait for a frame newer than number `after`. Returns (number, jpeg or None)."""
        with self._cond:
            self._cond.wait_for(lambda: self._count > after, timeout=timeout)
            return self._count, self._frame

    def mjpeg(self):
        """Multipart MJPEG for an <img> tag: every new frame, as it arrives."""
        seen = 0
        while not self._stopping.is_set():
            seen, frame = self.next_frame(seen)
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame)
                       + frame + b"\r\n")

    def capture(self):
        """The newest frame, sized for the model. Used by the scanner."""
        _, frame = self.next_frame(0, timeout=10)
        if frame is None:
            raise CameraError(self.error or "No picture from the camera yet.")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "mjpeg", "-i", "pipe:0",
               "-vf", "scale=w=%d:h=%d:force_original_aspect_ratio=decrease" % (self.look_size, self.look_size),
               "-frames:v", "1", "-f", "image2", "-vcodec", "mjpeg", "-q:v", "3", "pipe:1"]
        result = subprocess.run(cmd, input=frame, capture_output=True, timeout=15)
        if result.returncode != 0 or not result.stdout:
            raise CameraError("Couldn't resize the camera picture.")
        return result.stdout

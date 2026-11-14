"""OpenCV camera: USB webcams / USB microscopes, and network cameras (rtsp:// or http://).

What: `source: 0` means the first USB camera; a URL means a network camera. On each inspection
it throws away frames the camera had already buffered and returns a fresh one, so the picture
really shows the unit in front of it now (cameras keep several old frames in a queue).
Why: OpenCV can open all of these the same way, so one plugin covers most cameras you can
buy. If the camera is unplugged or the network drops, it reports the error and tries to
reconnect next time instead of crashing the app. Passwords inside a camera URL are hidden
(***) everywhere they could be shown or saved; only OpenCV gets the real URL.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
from PIL import Image

from .base import CameraError, Frame

TIMEOUT_MS = 5000
FLUSH_FRAMES = 4          # buffered frames to throw away before taking the picture
SECRET_KEYS = re.compile(r"(user|usr|login|pass|pwd|password|token|auth|key|secret|sig)", re.I)


def redact(source) -> str:
    """Hide passwords: rtsp://user:pw@host/x -> rtsp://***@host/x, ?pwd=x -> ?pwd=***."""
    if isinstance(source, int):
        return str(source)
    s = str(source)
    # 1) anything between "//" and "@" is user:password, whether or not the URL parses cleanly
    s = re.sub(r"//[^/@?#]*@", "//***@", s)
    try:
        u = urlsplit(s)
    except ValueError:
        return re.sub(r"\?.*$", "?***", s)
    # 2) query values whose names look like credentials
    if u.query:
        q = [(k, "***" if SECRET_KEYS.search(k) else v) for k, v in parse_qsl(u.query, keep_blank_values=True)]
        s = urlunsplit((u.scheme, u.netloc, u.path, urlencode(q, safe="*"), u.fragment))
    return s


class OpenCVCamera:
    def __init__(self, name: str, view: str, options: dict):
        self.name, self.view = name, view
        src = options.get("source", 0)
        self.source = int(src) if str(src).isdigit() else str(src)
        self.shown = redact(self.source)        # safe to display / store
        self.width, self.height = options.get("width"), options.get("height")
        self._cap = None
        self._error = ""

    def _open(self):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_MS]
        cap = cv2.VideoCapture(self.source, cv2.CAP_ANY, params)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"cannot open camera {self.shown}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # ask for a short queue (not all drivers obey)
        if self.width and self.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        return cap

    def capture(self, unit: int) -> Frame:
        try:
            if self._cap is None:
                self._cap = self._open()
            for _ in range(FLUSH_FRAMES):              # drop old buffered frames
                self._cap.grab()
            ok = self._cap.grab()
            ok, frame = self._cap.retrieve() if ok else (False, None)
            if not ok or frame is None:
                self._cap.release()
                self._cap = None
                raise CameraError(f"camera {self.shown} gave no picture")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except cv2.error as e:
            self._error = f"camera {self.shown}: {e}".splitlines()[0]
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            raise CameraError(self._error) from e
        except CameraError as e:
            self._error = str(e)
            raise
        self._error = ""
        return Frame(self.name, self.view, img, self.shown)

    def status(self) -> dict:
        return {"name": self.name, "type": "opencv", "view": self.view, "ok": not self._error,
                "detail": self._error or self.shown}

    def layout(self) -> dict | None:
        return None

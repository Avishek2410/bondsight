"""Folder camera: pretends to be a camera by reading photos from a folder, one per inspection.

What: each time it is asked for a picture it returns the next image file (in name order) that
it has not returned before. When none are left it says "no new image".
Why: lets you test the whole system with saved photos, and later lets another program (or a
real camera tool) drop images into a folder for inspection.
A file that cannot be read yet (another program is still writing it) is tried again next
time; only after 3 failed tries is it skipped as broken, so one bad file never blocks the rest.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..imageio import to_rgb8
from .base import CameraError, Frame

MAX_TRIES = 3


class FolderCamera:
    def __init__(self, name: str, view: str, options: dict):
        self.name, self.view = name, view
        self.path = Path(options["path"])
        self.patterns = [p.strip() for p in str(options.get("pattern", "*.jpg,*.jpeg,*.png")).split(",")]
        self._last: str | None = options.get("start_after")
        self._tries: dict[str, int] = {}
        self._error = ""

    def _files(self) -> list[Path]:
        seen: set[Path] = set()
        for pat in self.patterns:
            seen.update(p for p in self.path.glob(pat) if p.is_file())
        return sorted(seen, key=lambda p: p.name)

    def capture(self, unit: int) -> Frame:
        if not self.path.is_dir():
            self._error = f"folder not found: {self.path}"
            raise CameraError(self._error)
        self._error = ""
        for p in self._files():
            if self._last is not None and p.name <= self._last:
                continue
            try:
                with Image.open(p) as im:
                    img = to_rgb8(im)
                    img.load()
            except Exception as e:  # half-written file, corrupt file, decompression bomb, ...
                n = self._tries[p.name] = self._tries.get(p.name, 0) + 1
                if n >= MAX_TRIES:
                    self._last = p.name                 # give up on this file, move on next time
                    self._tries.pop(p.name, None)
                    raise CameraError(f"skipped unreadable {p.name} after {n} tries: {e}") from e
                raise CameraError(f"cannot read {p.name} yet (try {n}/{MAX_TRIES}): {e}") from e
            self._last = p.name
            self._tries.pop(p.name, None)
            return Frame(self.name, self.view, img, str(p))
        raise CameraError("no new image")

    def status(self) -> dict:
        ok = self.path.is_dir() and not self._error
        detail = self._error or (f"{self.path}" if ok else f"folder not found: {self.path}")
        return {"name": self.name, "type": "folder", "view": self.view, "ok": ok, "detail": detail}

    def layout(self) -> dict | None:
        return None

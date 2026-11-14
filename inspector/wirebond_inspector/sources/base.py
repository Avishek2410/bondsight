"""What every camera plugin looks like.

What: a "camera" is anything that can hand over one picture when asked: a real camera, a
folder of photos, or the synthetic generator. Each one has a name, a view (top or angled),
`capture(unit)` to take a picture, and `status()` to say whether it is working.
Why: the rest of the app never needs to know what kind of camera it is talking to, so adding
a new camera type means writing one small class, nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


class CameraError(RuntimeError):
    """This camera could not give a picture right now (the message says why)."""


@dataclass(frozen=True)
class Frame:
    camera: str
    view: str
    image: Image.Image
    source: str          # file path, URL, or "gen:u00012" for a synthetic unit


class Camera(Protocol):
    name: str
    view: str

    def capture(self, unit: int) -> Frame: ...

    def status(self) -> dict: ...

    def layout(self) -> dict | None: ...

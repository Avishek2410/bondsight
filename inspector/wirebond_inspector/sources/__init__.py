"""Camera plugins. `make_camera` picks the right class from the camera's `type` in config.yaml."""
from __future__ import annotations

from ..config import CAMERA_TYPES, CameraCfg, ConfigError
from .base import Camera, CameraError, Frame


def make_camera(cfg: CameraCfg) -> Camera:
    if cfg.type == "folder":
        from .folder import FolderCamera
        return FolderCamera(cfg.name, cfg.view, cfg.options)
    if cfg.type == "generator":
        from .generator import GeneratorCamera
        return GeneratorCamera(cfg.name, cfg.view, cfg.options)
    if cfg.type == "opencv":
        from .opencv_cam import OpenCVCamera
        return OpenCVCamera(cfg.name, cfg.view, cfg.options)
    raise ConfigError(f"camera {cfg.name!r}: unknown type {cfg.type!r} (use one of {', '.join(CAMERA_TYPES)})")


__all__ = ["Camera", "CameraError", "Frame", "make_camera"]

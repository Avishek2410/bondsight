"""Generator camera: draws a brand-new synthetic wire-bond chip for every inspection.

What: uses our own generator (wirebond_synth) to render "unit number N" of a product, with
defects injected the way the config says (random, or one wire that keeps failing).
Why: we have no real camera attached yet. Two generator cameras with the same product_seed
(one `top`, one `angled`) show the SAME chip from two angles, exactly like the two real
cameras will. Because we know which defects were drawn, the UI can show the true answer
next to what the AI found.
"""
from __future__ import annotations

from wirebond_synth.augment import augment
from wirebond_synth.dataset import unit_seed
from wirebond_synth.defects import DefectSpec, InjectionConfig, plan_unit_defects
from wirebond_synth.layout import make_layout
from wirebond_synth.render import render_unit
from PIL import Image

from .base import Frame


class GeneratorCamera:
    def __init__(self, name: str, view: str, options: dict):
        self.name, self.view = name, view
        self.seed = int(options.get("product_seed", 7))
        self.size = int(options.get("size", 640))
        self.strength = float(options.get("augment_strength", 1.0))
        self._layout = make_layout(int(options.get("n_wires", 24)), self.seed, self.size)
        self.cfg = InjectionConfig(**(options.get("injection") or {"mode": "random", "p": 0.05}))
        plan_unit_defects(self._layout, 0, self.cfg, self.seed)       # fail early on a bad config

    def planned_defects(self, unit: int) -> list[DefectSpec]:
        return plan_unit_defects(self._layout, unit, self.cfg, self.seed)

    def capture(self, unit: int) -> Frame:
        # same seeds as wirebond_synth.dataset.generate, so results match a generated dataset
        views = ("top", "angled")
        img, boxes = render_unit(self._layout, self.planned_defects(unit), self.view,
                                 unit_seed(self.seed, unit))
        img, _ = augment(img, boxes, unit_seed(self.seed, unit) * 2 + views.index(self.view),
                         self.strength)
        rgb = Image.fromarray(img[:, :, ::-1].copy())                  # OpenCV BGR -> RGB
        return Frame(self.name, self.view, rgb, f"gen:u{unit:05d}")

    def status(self) -> dict:
        return {"name": self.name, "type": "generator", "view": self.view, "ok": True,
                "detail": f"synthetic product {self.seed}, {len(self._layout.wires)} wires, {self.cfg.mode}"}

    def layout(self) -> dict | None:
        return self._layout.to_dict()

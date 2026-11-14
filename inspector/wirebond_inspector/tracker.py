"""Spot repetitive defects: the same defect at the same wire, again and again.

What: keeps the last M units in memory. After each unit, for every (position, defect class)
found in THIS unit, it counts how many of the last M units had it too. If that is K or more,
it raises an alert, e.g. "wire:5 wire_sweep in 3 of the last 10 units".
Why: a random defect happens now and then. A repetitive one means something is wrong with the
machine (a worn bonding tool, a bad bond program) and will keep producing scrap until fixed.
That is the most valuable thing an inspector can tell an engineer. Defaults K=3, M=10.
"""
from __future__ import annotations

from collections import deque


class RepetitiveTracker:
    def __init__(self, k: int = 3, m: int = 10):
        self.k, self.m = k, m
        self._window: deque[set[tuple[str, str]]] = deque(maxlen=m)

    def load(self, history: list[set[tuple[str, str]]]) -> None:
        """Refill from the database after a restart (oldest first)."""
        self._window.clear()
        for h in history[-self.m:]:
            self._window.append(set(h))

    def update(self, unit_findings: set[tuple[str, str]]) -> list[dict]:
        self._window.append(set(unit_findings))
        alerts = []
        for pos, cls in sorted(unit_findings):
            n = sum((pos, cls) in u for u in self._window)
            if n >= self.k:
                alerts.append({"position": pos, "cls": cls, "count": n, "window": self.m})
        return alerts

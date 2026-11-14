import numpy as np
import pytest

from wirebond_synth.defects import DefectSpec, sample_params
from wirebond_synth.layout import make_layout


@pytest.fixture(scope="session")
def L():
    return make_layout(16, seed=1)


def plan_one(cls: str, wire: int = 4, extreme: bool = False) -> DefectSpec:
    """A valid DefectSpec for class `cls` on `wire` (max-range params if extreme)."""
    return DefectSpec(wire, cls, sample_params(cls, np.random.default_rng(0), extreme=extreme))

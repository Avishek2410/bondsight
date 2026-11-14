"""The fixed defect-class table. Order matters: COCO category id = index + 1."""

CLASSES: tuple[str, ...] = (
    "wire_sweep",       # loop bowed sideways (mould-flow drag)
    "lifted_ball",      # 1st bond not welded to the pad
    "lifted_stitch",    # 2nd bond not welded to the lead
    "missing_wire",     # no wire at all
    "broken_wire",      # wire has a gap
    "off_center_ball",  # ball not centred on its pad (capillary wear)
    "wire_short",       # wire touching its neighbour
    "sagging_loop",     # loop too low, near the die edge
)

VIEWS: tuple[str, ...] = ("top", "angled")

# Which camera view can see each defect. Height-only defects need the angled view.
VISIBILITY: dict[str, frozenset[str]] = {
    c: frozenset({"angled"}) if c in ("lifted_stitch", "sagging_loop") else frozenset(VIEWS)
    for c in CLASSES
}


def class_id(name: str) -> int:
    """1-based COCO category id of a class name."""
    return CLASSES.index(name) + 1

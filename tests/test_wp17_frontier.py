"""Small CPU frontier cases; no material stepping or historical physics suite.

The launch fixtures include every argument of the original v7_2 kernel. Both
paths receive identical inputs and must retain the explicit expected codes,
pending evidence, diagnostic counts and read-only arrays.
"""
import numpy as np
import pytest
import warp as wp

from robotic_cutting_sim import tlmpm_v7_2 as C
from robotic_cutting_sim import tlmpm_v7_2_frontier as F
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V2_Y


_NAMES = (
    "start", "x", "velocity", "mass", "old_codes", "codes", "reached", "nodes",
    "weights", "gradients", "offsets", "neighbours", "poses", "previous",
    "twists", "cuts", "index", "stroke", "segment_ok", "segments", "width",
    "band", "bevel", "half_thickness", "profile", "profile_count", "radius",
    "dt", "friction", "slice_efficiency", "pending_q", "pending_cut", "diagnostics",
)
_DTYPES = {
    **dict.fromkeys(("start", "x", "velocity", "gradients"), wp.vec3),
    **dict.fromkeys(("mass", "weights"), float),
    **dict.fromkeys(("old_codes", "codes", "reached", "nodes", "offsets",
                     "neighbours", "cuts", "index", "stroke", "segment_ok",
                     "pending_q", "pending_cut", "diagnostics"), int),
    "poses": wp.transform, "previous": wp.transform,
    "twists": wp.spatial_vector, "profile": wp.vec2,
}
_OUTPUTS = ("codes", "pending_q", "pending_cut", "diagnostics")


def _fixture(case):
    """Two/three particles and an explicit reference-node neighbour order."""
    count = 3 if case.startswith(("first_", "node_first_")) else 2
    positions = np.array([[.0001, 0., -.0002]] + [[.0001, 0., .0002]]*(count-1), np.float32)
    polygon, _ = RULER_KNIFE_V2_Y.tlmpm_profile()
    data = dict(
        start=positions.copy(), x=positions.copy(), velocity=np.zeros((count, 3), np.float32),
        mass=np.full(count, 1.5e-6, np.float32), old_codes=np.zeros(count, np.int32),
        codes=np.zeros(count, np.int32), reached=np.array([1]+[0]*(count-1), np.int32),
        nodes=np.zeros((count, 8), np.int32), weights=np.ones((count, 8), np.float32),
        gradients=np.zeros((count, 8, 3), np.float32),
        offsets=np.array([0, count], np.int32), neighbours=np.arange(count, dtype=np.int32),
        poses=np.array([[.0002, 0., 0., 0., 0., 0., 1.]], np.float32),
        previous=np.array([[0., 0., 0., 0., 0., 0., 1.]], np.float32),
        twists=np.array([[.01, 0., 0., 0., 0., 0.]], np.float32),
        cuts=np.array([0], np.int32), index=np.array([0], np.int32),
        stroke=np.array([1], np.int32), segment_ok=np.array([1, 1], np.int32),
        segments=2, width=.06, band=.024, bevel=.003383, half_thickness=.00075,
        profile=np.asarray(polygon, np.float32), profile_count=len(polygon),
        radius=.00045, dt=.0001, friction=.3, slice_efficiency=1.,
        pending_q=np.full(count, -1, np.int32), pending_cut=np.full(count, -1, np.int32),
        diagnostics=np.zeros(2, np.int32),
    )
    if case.startswith("pending_") or case in ("already_assigned", "unreached"):
        data["pending_q"][0], data["pending_cut"][0] = 1, 0
    if case in ("no_contact", "first_unworked", "first_worked", "node_first_unworked", "node_first_worked"):
        data["start"][1, 2] = data["x"][1, 2] = .01
    if case == "first_worked":
        data["neighbours"] = np.array([0, 2, 1], np.int32)
    if case.startswith("node_first_"):
        data["nodes"][0, 1::2] = 1
        data["offsets"] = np.array([0, 2, 4], np.int32)
        data["neighbours"] = np.array([0, 1, 0, 2], np.int32)
        if case == "node_first_worked":
            data["nodes"][0] = 1-data["nodes"][0]
    if case in ("no_sweep", "pending_reuse", "pending_stale_cut", "pending_stopped",
                "pending_segment_blocked", "pending_empty_keeps_cut"):
        data["previous"] = data["poses"].copy()
    if case == "pending_empty_keeps_cut":
        data["pending_q"][0] = -1
    if case in ("pending_disarmed",):
        data["cuts"][0] = -1
    if case == "no_stroke":
        data["stroke"][0] = 0
    if case == "pending_stopped":
        data["twists"][:] = 0.
    if case in ("small_rotation", "pending_large_rotation"):
        omega = .01 if case == "small_rotation" else 2.
        angle = omega*data["dt"]
        data["poses"][0, 5:] = [np.sin(angle/2), np.cos(angle/2)]
        data["twists"][0, 5] = omega
    if case in ("outside_edge", "pending_outside_edge"):
        data["start"][:, 1] = data["x"][:, 1] = .1
    if case == "pending_band_exit":
        data["band"] = .00005
    if case == "pending_side_exit":
        data["reached"][0] = 2
    if case == "pending_bevel_exit":
        data["poses"][0, 0] = .004
    if case == "pending_lost_interface":
        data["start"][1, 2] = data["x"][1, 2] = -.0002
    if case == "pending_stale_cut":
        data["pending_cut"][0] = 1
    if case == "already_assigned":
        data["old_codes"][0] = data["codes"][0] = 1
    if case == "unreached":
        data["reached"][0] = 0
    if case in ("zero_weight_gradient", "zero_weight_no_gradient"):
        data["weights"][:] = 0.
        if case == "zero_weight_gradient":
            data["gradients"][0, 0, 0] = 1.
    if case == "duplicate_shared_nodes":
        data["nodes"][:, 1::2] = 1
        data["offsets"] = np.array([0, 2, 4], np.int32)
        data["neighbours"] = np.array([0, 1, 0, 1], np.int32)
    if case in ("code_zero_bridge", "conflicting_history", "second_cut"):
        data["old_codes"][0] = data["codes"][0] = 1 if case == "second_cut" else 4
        if case == "conflicting_history":
            data["old_codes"][1] = data["codes"][1] = 8
        if case == "second_cut":
            data["cuts"][0], data["stroke"][0], data["reached"][0] = 1, 2, 4
    if case in ("segment_boundary_allowed", "segment_boundary_blocked", "segment_left_allowed",
                "pending_segment_blocked"):
        data["segment_ok"][:] = [0, 1] if case == "segment_boundary_allowed" else [1, 0]
        if case == "segment_left_allowed":
            data["start"][:, 1] = data["x"][:, 1] = -.00001
        if case == "pending_segment_blocked":
            data["segment_ok"][:] = 0
    if case in ("crossing_segment_only", "current_segment_only"):
        sign = 1. if case == "crossing_segment_only" else -1.
        data["start"][:, 1], data["x"][:, 1] = -.0002*sign, .00005*sign
        data["segment_ok"][:] = [1, 0]
    if case in ("bevel_exact", "bevel_outside"):
        data["start"][:, 0] = data["x"][:, 0] = 0.
        data["previous"][0, 0] = -.0001
        data["poses"][0, 0] = np.float32(data["bevel"])
        if case == "bevel_outside":
            data["poses"][0, 0] = np.nextafter(data["poses"][0, 0], np.float32(np.inf))
    if case in ("width_positive", "width_negative", "width_outside"):
        y = np.float32(-data["width"] if case == "width_negative" else data["width"])
        if case == "width_outside":
            y = np.nextafter(y, np.float32(np.inf))
        data["start"][:, 1] = data["x"][:, 1] = y
    if case in ("old_on_edge", "both_on_edge"):
        data["previous"][0, 0] = data["start"][0, 0]
    if case in ("new_on_edge", "both_on_edge"):
        data["poses"][0, 0] = data["x"][0, 0]
    if case in ("tiny_opposite_sides", "tiny_same_side", "zero_side"):
        data["start"][0, 2] = data["x"][0, 2] = -1.e-12
        height = -1.e-12 if case == "tiny_same_side" else (0. if case == "zero_side" else 1.e-12)
        data["start"][1, 2] = data["x"][1, 2] = height
    if case == "underflow_opposite_sides":
        # Preserve sa*sb < 0 itself: opposite signs are insufficient when the
        # float32 product underflows to negative zero.
        data["start"][:, 2] = data["x"][:, 2] = [-1.e-25, 1.e-25]
    return data


def _arguments(data):
    return [wp.array(data[name], dtype=_DTYPES[name], device="cpu") if name in _DTYPES else data[name]
            for name in _NAMES]


def _launch(kernel, arguments):
    readonly = {name: arguments[i].numpy().copy() for i, name in enumerate(_NAMES)
                if name in _DTYPES and name not in _OUTPUTS}
    wp.launch(kernel, len(arguments[0]), arguments, device="cpu")
    for name, before in readonly.items():
        np.testing.assert_array_equal(arguments[_NAMES.index(name)].numpy(), before, err_msg=name)
    return {name: arguments[_NAMES.index(name)].numpy().copy() for name in _OUTPUTS}


_CASES = [
    # case, first particle code, pending neighbour, new crossings/completions
    ("contact", 1, -1, (1, 1)),
    ("no_contact", 0, 1, (1, 0)),
    ("first_unworked", 0, 1, (1, 0)),
    ("first_worked", 1, -1, (1, 1)),
    ("node_first_unworked", 0, 1, (1, 0)),
    ("node_first_worked", 1, -1, (1, 1)),
    ("no_sweep", 0, -1, (0, 0)),
    ("no_stroke", 0, 1, (1, 0)),
    ("pending_reuse", 1, -1, (0, 1)),
    ("pending_stopped", 0, 1, (0, 0)),
    ("pending_segment_blocked", 0, 1, (0, 0)),
    ("pending_stale_cut", 0, -1, (0, 0)),
    ("pending_empty_keeps_cut", 0, -1, (0, 0)),
    ("pending_disarmed", 0, -1, (0, 0)),
    ("already_assigned", 1, -1, (0, 0)),
    ("unreached", 0, -1, (0, 0)),
    ("small_rotation", 1, -1, (1, 1)),
    ("pending_large_rotation", 0, -1, (0, 0)),
    ("outside_edge", 0, -1, (0, 0)),
    ("pending_outside_edge", 0, -1, (0, 0)),
    ("pending_band_exit", 0, -1, (0, 0)),
    ("pending_side_exit", 0, -1, (0, 0)),
    ("pending_bevel_exit", 0, -1, (0, 0)),
    ("pending_lost_interface", 0, -1, (0, 0)),
    ("zero_weight_gradient", 1, -1, (1, 1)),
    ("zero_weight_no_gradient", 0, -1, (0, 0)),
    ("duplicate_shared_nodes", 1, -1, (1, 1)),
    ("code_zero_bridge", 5, -1, (1, 1)),
    ("conflicting_history", 4, -1, (0, 0)),
    ("second_cut", 5, -1, (1, 1)),
    ("segment_boundary_allowed", 1, -1, (1, 1)),
    ("segment_boundary_blocked", 0, -1, (0, 0)),
    ("segment_left_allowed", 1, -1, (1, 1)),
    ("crossing_segment_only", 0, 1, (1, 0)),
    ("current_segment_only", 0, -1, (0, 0)),
    ("bevel_exact", 1, -1, (1, 1)),
    ("bevel_outside", 0, -1, (0, 0)),
    ("width_positive", 1, -1, (1, 1)),
    ("width_negative", 1, -1, (1, 1)),
    ("width_outside", 0, -1, (0, 0)),
    ("old_on_edge", 1, -1, (1, 1)),
    ("new_on_edge", 1, -1, (1, 1)),
    ("both_on_edge", 0, -1, (0, 0)),
    ("tiny_opposite_sides", 1, -1, (1, 1)),
    ("underflow_opposite_sides", 0, -1, (0, 0)),
    ("tiny_same_side", 0, -1, (0, 0)),
    ("zero_side", 0, -1, (0, 0)),
]


@pytest.mark.parametrize("case,code,pending,counts", _CASES, ids=[row[0] for row in _CASES])
def test_frontier_matches_original_and_expected_state(case, code, pending, counts):
    data = _fixture(case)
    actual = _launch(C._complete_links, _arguments(data))
    optimized = _launch(F._complete_links, _arguments(data))
    for name in _OUTPUTS:
        np.testing.assert_array_equal(optimized[name], actual[name], err_msg=name)
    expected_codes = data["old_codes"].copy()
    expected_codes[0] = code
    np.testing.assert_array_equal(actual["codes"], expected_codes)
    expected_pending = np.full(len(expected_codes), -1, np.int32)
    expected_pending[0] = pending
    np.testing.assert_array_equal(actual["pending_q"], expected_pending)
    expected_pending_cut = np.full(len(expected_codes), -1, np.int32)
    if pending >= 0 or case == "pending_empty_keeps_cut":
        expected_pending_cut[0] = data["cuts"][0]
    np.testing.assert_array_equal(actual["pending_cut"], expected_pending_cut)
    np.testing.assert_array_equal(actual["diagnostics"], counts)


def _pending_lifecycle(kernel, case):
    data = _fixture(case)
    arguments = _arguments(data)
    first = _launch(kernel, arguments)
    restored = _fixture("contact")
    # No new sweep on the second launch: only genuinely retained evidence may complete.
    restored["previous"] = restored["poses"].copy()
    for name in ("start", "x", "previous", "poses", "twists", "cuts", "reached", "segment_ok"):
        arguments[_NAMES.index(name)].assign(restored[name])
    arguments[_NAMES.index("band")] = restored["band"]
    second = _launch(kernel, arguments)
    return first, second


@pytest.mark.parametrize("case", ["no_contact", "pending_disarmed", "pending_large_rotation",
                                  "pending_outside_edge", "pending_band_exit", "pending_side_exit",
                                  "pending_bevel_exit", "pending_lost_interface"])
def test_frontier_pending_lifecycle_matches_original(case):
    first, second = _pending_lifecycle(C._complete_links, case)
    optimized = _pending_lifecycle(F._complete_links, case)
    for reference, candidate in zip((first, second), optimized):
        for name in _OUTPUTS:
            np.testing.assert_array_equal(candidate[name], reference[name], err_msg=name)
    retained = case == "no_contact"
    assert first["pending_q"].tolist() == ([1, -1] if retained else [-1, -1])
    assert second["codes"].tolist() == ([1, 0] if retained else [0, 0])
    assert second["pending_q"].tolist() == second["pending_cut"].tolist() == [-1, -1]
    assert second["diagnostics"].tolist() == ([1, 1] if retained else [0, 0])


def test_frontier_solver_default_and_original_rollback(monkeypatch):
    import hashlib
    from pathlib import Path
    from robotic_cutting_sim.blade_probe import block
    from robotic_cutting_sim.material_config import TLMPMConfig

    points, masses = block(size_m=(.016, .008, .008), spacing=.004)
    config = TLMPMConfig(cuts=1, floor=False, fixture_walls=False, tool=False,
                         material_contact=False, gravity=(0., 0., 0.))
    beds = [C.TLMPM(points, masses, .004, config, "cpu", optimize_frontier=False),
            C.TLMPM(points, masses, .004, config, "cpu")]
    launched = []
    original_launch = wp.launch

    def record_launch(kernel, *args, **kwargs):
        if kernel in (C._complete_links, F._complete_links):
            launched.append(kernel)
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", record_launch)
    for bed in beds:
        bed.step(advance=False)
    assert launched == [C._complete_links, F._complete_links]
    for bed, enabled in zip(beds, (False, True)):
        receipt = bed.solver_receipt()
        assert receipt["optimize_frontier"] is enabled
        assert receipt["frontier_kernel"] == (F.__name__ if enabled else C.__name__)+"._complete_links"
        assert receipt["frontier_execution"] == ("hoisted_interface_v1" if enabled else "original_v7_2")
        assert receipt["frontier_source_sha256"] == (hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest()
                                                       if enabled else None)
    for name in ("positions", "velocities", "F", "Fpi", "codes", "contact_work_j"):
        np.testing.assert_array_equal(beds[1].snapshot()[name], beds[0].snapshot()[name], err_msg=name)


@pytest.mark.parametrize("enabled", [False, True])
def test_existing_live_switch_restores_both_original_paths(monkeypatch, enabled):
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial

    captured = {}

    def stop_before_allocation(self, *args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop_before_solver_allocation")

    monkeypatch.setattr(C.TLMPM, "__init__", stop_before_allocation)
    material = LiveCuttingMaterial(CouplingConfig(device="cpu"), solver="tlmpm_v7_2",
                                   tool_profile=RULER_KNIFE_V2_Y, cutting_law=8,
                                   v7_2_optimizations=enabled)
    with pytest.raises(RuntimeError, match="stop_before_solver_allocation"):
        material.reset([0., 0., 0., 0., 0., 0., 1.], seed=42)
    overrides = {key: captured[key] for key in ("optimize_contacts", "optimize_frontier") if key in captured}
    assert overrides == ({} if enabled else {"optimize_contacts": False, "optimize_frontier": False})

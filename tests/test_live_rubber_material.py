"""R1.4: LiveRubberMaterial (CPU path), index continuity, records, validity flags, reset."""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_phys as P
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.live_rubber_material import (INVALID_RUBBER_J_DOMAIN, LiveRubberMaterial,
                                                      RubberInvalidState, validate_records)


def _cantilever(**kw):
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2)
    return LiveRubberMaterial(spec, material=P.material(4e5), control_dt=2e-4, **kw)


def test_step_rule_integer_substeps_and_receipt():
    live = _cantilever()
    assert live.substeps*live.dt == pytest.approx(live.control_dt, rel=1e-12)
    assert live.step_rule["cfl_actual"] <= 0.5
    rc = live.receipt()
    assert rc["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED" and rc["graph"] is None
    assert rc["solver"]["pic"] == 0.0 and rc["solver"]["status"] == "EXPERIMENTAL_NOT_ACCEPTED"


def test_chunked_substeps_equal_one_control_step():
    """Index continuity: K substeps split into chunks (as the CUDA graph replays) equal the advance() loop."""
    a, b = _cantilever(), _cantilever()
    move = {0: {"translation": np.array([0.0, 0.0, 1e-4])}}
    a.advance(move)
    s = b.solver
    s.begin_control_step()
    s.set_group_motion(0, translation=np.array([0.0, 0.0, 1e-4]))
    for chunk in (b.substeps//3, b.substeps//3, b.substeps-2*(b.substeps//3)):
        for _ in range(chunk):
            s.step()
    for name in ("x", "v", "F", "stress", "new_v", "reaction", "index"):
        assert np.array_equal(getattr(a.solver, name).numpy(), getattr(s, name).numpy()), name
    assert int(s.index.numpy()[0]) == b.substeps


def test_motion_ends_exactly_at_target_and_holds():
    live = _cantilever(gravity=(0, 0, 0))
    d = np.array([0.0, 0.0, 2e-4])
    live.advance({0: {"translation": d}})
    jaw = np.flatnonzero(live.scene.X[:, 0] < -live.scene.dx*1.2)
    disp = live.solver.x.numpy()[jaw].astype(np.float64)-live.scene.X[jaw]
    assert np.abs(disp-d).max() < 1e-8
    live.advance()          # unlisted group holds at the previous end
    disp2 = live.solver.x.numpy()[jaw].astype(np.float64)-live.scene.X[jaw]
    assert np.abs(disp2-d).max() < 1e-8


def test_records_schema_and_reset():
    live = _cantilever(record=True, gravity=(0, 0, -9.81))
    for _ in range(3):
        live.advance()
    assert validate_records(live.records, expected_steps=3) == []
    bad = [dict(r) for r in live.records]
    bad[2] = {k: v for k, v in bad[2].items() if k != "energies_j"}
    assert validate_records(bad)
    live.reset()
    assert len(live.records) == 1 and live.step_count == 0
    assert np.array_equal(live.solver.x.numpy(), live.scene.X)
    assert np.abs(live.solver.v.numpy()).max() == 0.0


def test_j_domain_violation_strict_and_sticky():
    spec = rs.RubberStripSpec(length=0.004, width=0.004, thickness=0.004, dx=0.001, case="block")
    live = LiveRubberMaterial(spec, material=P.material(4e5), control_dt=1e-4, gravity=(0, 0, 0))
    with pytest.raises(RubberInvalidState, match="RUBBER_J_DOMAIN"):
        live.advance({0: {"affine": np.diag([0.5, 0.5, 0.5])}})          # J = 0.125 < 0.2
    soft = LiveRubberMaterial(spec, material=P.material(4e5), control_dt=1e-4, gravity=(0, 0, 0), strict=False,
                              record=True)
    soft.advance({0: {"affine": np.diag([0.5, 0.5, 0.5])}})
    soft.advance({0: {"affine": np.eye(3)}})                             # back inside: the flag stays (sticky)
    assert soft.invalid_reason & INVALID_RUBBER_J_DOMAIN
    assert soft.records[-1]["invalid_flags"] == ["RUBBER_J_DOMAIN"]
    assert np.isfinite(soft.solver.stress.numpy()).all()                 # clamped only to stay finite
    soft.reset()
    assert soft.invalid_reason == 0 and int(soft.solver.violations.numpy()[0]) == 0


def test_graph_capture_needs_cuda():
    with pytest.raises(ValueError, match="CUDA"):
        _cantilever(capture_graph=True)

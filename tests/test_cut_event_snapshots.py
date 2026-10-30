"""cut_event_snapshots_v1 and drive_blade_sequence (WP-12 Phase 1; Warp on "cpu", coarse 4 mm blocks).

- the diagnostic recorder is read-only: a law-8 cut with it equals the run with the plain cut_events_v1
  recorder and the run without any recorder, bit for bit (every snapshot array);
- its log equals the parent's, and every assignment-instant Hermite point lies on the solver's swept plane
  (o, n) of its substep; assignment substeps equal the logged substeps;
- ``installed`` restores the module attribute; the chunk hook sees every chunk;
- drive_blade_sequence with one constant orientation reproduces drive_blade bit for bit;
- surface_attribution_v1.stencil equals the tlmpm_v5 reference stencil.
"""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from robotic_cutting_sim import cut_events as ce
from robotic_cutting_sim import cut_event_snapshots as cs
from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block, drive_blade
from robotic_cutting_sim.sequence_blade_probe import OrientedPhase, drive_blade_sequence
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1
from scipy.spatial.transform import Rotation

TOP, SPACING = .012, .004
ROT = Rotation.from_euler("z", 20, degrees=True)*VERTICAL_CUT
PHASES = [Phase((0, 0, -.02), .9, 0), Phase((0, 0, .05), .2, -1)]
START = [.001, 0, TOP+.004]


def probe(**kwargs):
    return drive_blade(RULER_KNIFE_V1, ROT, START, PHASES, cutting_law=8, solver_version=5, device="cpu", **kwargs)


@pytest.fixture(scope="module")
def runs():
    calls = []
    with cs.installed(hook=lambda rec: calls.append(int(rec.substep.numpy()[0]))) as made:
        diag = probe(cut_events="cut_events_v1")
    assert ce.CutEventRecorder is not made[0].__class__ and not issubclass(ce.CutEventRecorder, cs.AssignmentSnapshotRecorder)
    plain = probe(cut_events="cut_events_v1")
    off = probe()
    return diag, plain, off, made[0], calls


def test_diagnostic_recorder_is_physics_neutral(runs):
    diag, plain, off, _, _ = runs
    for key in off["state"]:
        assert np.array_equal(diag["state"][key], off["state"][key]), key
        assert np.array_equal(plain["state"][key], off["state"][key]), key
    assert diag["labelled_after_phase"] == off["labelled_after_phase"]


def test_log_equals_parent_and_points_lie_on_the_swept_plane(runs):
    diag, plain, _, rec, calls = runs
    log = diag["cut_events"]["log"]
    assert len(log) > 0 and np.array_equal(log, plain["cut_events"]["log"])
    a = rec.assignment_positions(log)
    assert a["substep_matches_log"]
    d = np.einsum("ij,ij->i", a["hermite_current_at_assignment"]-log["edge_origin"], log["normal_current"])
    assert np.abs(d).max() < 1e-6
    total = sum(int(round(p.duration_s/1e-4)) for p in PHASES)
    assert calls[-1] == total and len(calls) == sum(-(-int(round(p.duration_s/1e-4))//200) for p in PHASES)


def test_sequence_probe_reproduces_drive_blade():
    phases = [OrientedPhase(p.velocity_m_s, p.duration_s, p.cut, tuple(ROT.as_quat())) for p in PHASES]
    seq = drive_blade_sequence(RULER_KNIFE_V1, START, phases, device="cpu", cut_events="cut_events_v1")
    ref = probe(cut_events="cut_events_v1")
    for key in ref["state"]:
        assert np.array_equal(seq["state"][key], ref["state"][key]), key
    assert np.array_equal(seq["cut_events"]["log"], ref["cut_events"]["log"])
    assert seq["components"] == ref["components"]


def test_numpy_stencil_equals_tlmpm_v5():
    from robotic_cutting_sim import surface_attribution_v1 as A
    from robotic_cutting_sim.material_config import TLMPMConfig
    from robotic_cutting_sim.tlmpm_v5 import TLMPM
    points, masses = block()
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    bed = TLMPM(points, masses, SPACING, TLMPMConfig(cuts=1, blade_half_width=float(half)), "cpu",
                blade_profile_xz=polygon, cutting_law=8)
    nodes, weights, gradients = A.stencil(points, bed.config.dx)
    assert np.array_equal(nodes, bed.nodes.numpy()) and np.array_equal(weights, bed.weights.numpy())
    assert np.array_equal(gradients, bed.gradients.numpy())

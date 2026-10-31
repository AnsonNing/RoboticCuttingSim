"""WP-12 Phase 4 root-cause verification: physics-neutral recorders and the COUNTERFACTUAL tlmpm_v5 variant.

CPU checks (Warp device "cpu", 4 mm blocks of 270 particles, law 8):
- ``flip_drift_v1``, ``tool_contact_split_v1`` and ``tensile_energy_max_v1`` are physics-neutral: a kinematic law-8 cut with both recorders
  on equals the run with them off bit for bit (every snapshot array, reaction, stats, codes);
- the drift identity x = X + ugrid + drift holds to float32 rounding; first-assignment substeps are recorded for
  exactly the labelled particles;
- the split recorder reproduces the solver's blade reaction (in-zone + out-of-zone = solver reaction without
  obstacles) to float32 summation tolerance, finds in-zone contacts in an armed cut and none while disarmed;
- ``CounterfactualTLMPM`` with default options is bitwise equal to tlmpm_v5 (its step body is a copy); each
  counterfactual option changes the run; ``grid_musl`` makes x = X + ugrid (no drift in the positions);
- the counterfactual module is not named by any other module under ``src/`` (never selectable from production).
"""
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from robotic_cutting_sim import root_cause_recorders as rr
from robotic_cutting_sim import tlmpm_v5
from robotic_cutting_sim.blade_probe import VERTICAL_CUT, Phase, block
from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.tlmpm_v5_counterfactual import (COUNTERFACTUAL_LABEL, CounterfactualOptions,
                                                         CounterfactualTLMPM)
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012
X0 = .002
PHASES = [Phase((0, 0, -.05), .30, -1), Phase((0, 0, -.05), .26, 0), Phase((0, 0, .05), .04, -1)]
START = [X0, 0, TOP+.004]
SNAPSHOT_KEYS = ("positions", "velocities", "codes", "F", "Fpi", "plastic_strain", "cut_progress", "contact_work_j")


def make_bed(cls=tlmpm_v5.TLMPM, floor=True, **kwargs):
    """floor=True: legacy tray (blade_probe default); floor=False: a board fixture box under the block plus a
    table plane (the resolution / E1 support path)."""
    points, masses = block()
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    config = TLMPMConfig(cuts=1, blade_half_width=float(half), cutting_energy=5., floor=floor, fixture_walls=floor)
    if not floor:
        kwargs = {"fixture_boxes": [((0., 0., -.005), (.06, .04, .005))], "table_plane": [0., 0., 1., .010], **kwargs}
    return cls(points, masses, .004, config, "cpu", blade_profile_xz=polygon, cutting_law=8, **kwargs), points


def run(bed, drift=None, split=None, phases=PHASES, energy=None):
    quaternion, position = VERTICAL_CUT.as_quat(), np.asarray(START, float)
    reactions, stats, solver_blade = [], [], []
    dt = bed.config.dt
    for phase in phases:
        velocity = np.asarray(phase.velocity_m_s, float)
        steps = int(round(phase.duration_s/dt))
        for first in range(0, steps, 200):
            count = min(200, steps-first)
            poses = np.array([np.r_[position+velocity*dt*i, quaternion] for i in range(count+1)], np.float32)
            twists = np.tile(np.r_[velocity, 0., 0., 0.], (count+1, 1)).astype(np.float32)
            bed.set_trajectory(poses, twists, np.full(count+1, phase.cut, np.int32))
            for _ in range(count):
                bed.reaction.zero_()
                bed.stats.zero_()
                if split is not None:
                    split.before_step()
                bed.step()
                if drift is not None:
                    drift.after_step()
                if energy is not None:
                    energy.after_step()
                reactions.append(bed.reaction.numpy()[0].copy())
                stats.append(bed.stats.numpy().copy())
            position = position+velocity*dt*count
    return np.asarray(reactions), np.asarray(stats)


def assert_bitwise(a, b):
    sa, sb = a.snapshot(), b.snapshot()
    for key in SNAPSHOT_KEYS:
        assert sa[key].tobytes() == sb[key].tobytes(), key


@pytest.fixture(scope="module")
def baseline():
    bed, points = make_bed()
    reactions, stats = run(bed)
    return bed, points, reactions, stats


@pytest.mark.cpu_sim
def test_recorders_are_physics_neutral_bitwise(baseline):
    off, _, r_off, s_off = baseline
    on, _ = make_bed()
    drift = rr.FlipDriftRecorder(on)
    split = rr.ToolContactSplitRecorder(on, capacity=10000)
    energy = rr.TensileEnergyRecorder(on)
    r_on, s_on = run(on, drift, split, energy=energy)
    assert_bitwise(off, on)
    assert r_off.tobytes() == r_on.tobytes() and s_off.tobytes() == s_on.tobytes()
    assert np.count_nonzero(on.code.numpy()) > 0          # the run labels (neutrality covers assignment)
    assert drift.metadata()["substeps"] == len(r_on) == split.check()["substeps"]
    e = energy.arrays()
    assert (e["psi_tensile_max_j_m3"] > 0).any() and np.all(e["psi_total_max_j_m3"] >= e["psi_tensile_max_j_m3"]-1e-6)


@pytest.mark.cpu_sim
def test_drift_identity_and_assignment_record(baseline):
    bed, points = make_bed()
    drift = rr.FlipDriftRecorder(bed)
    run(bed, drift)
    a = drift.arrays()
    x = bed.x.numpy().astype(np.float64)
    residual = x-points.astype(np.float64)-a["ugrid_m"]-a["drift_m"]
    assert np.abs(residual).max() < 1e-6                  # float32 rounding of x over 6000 substeps (< 3e-4 s)
    assert np.abs(a["drift_m"]).max() > 1e-6              # the drift is real, not zero
    codes = bed.code.numpy()
    first = a["assign_substep"][:, 0]
    assert np.array_equal(first >= 0, codes != 0)
    assert first[codes != 0].min() >= int(round(.30/bed.config.dt))   # only after the armed phase starts


@pytest.mark.cpu_sim
def test_split_reproduces_blade_reaction():
    bed, _ = make_bed()                                    # no obstacle boxes: reaction is the blade alone
    split = rr.ToolContactSplitRecorder(bed, capacity=10000)
    reactions, _ = run(bed, split=split)
    t = split.table()
    col = rr.SPLIT_COLUMNS.index
    total_d = t[:, col("in_d")]+t[:, col("out_d")]
    # advance d = -z for VERTICAL_CUT; reaction force on the blade along d
    scale = 1e-5*np.abs(reactions[:, :3]).max()            # float32 summation-order tolerance
    np.testing.assert_allclose(total_d, -reactions[:, 2], rtol=1e-4, atol=scale)
    total_n = t[:, col("in_n")]+t[:, col("out_n")]
    np.testing.assert_allclose(total_n, reactions[:, 0], rtol=1e-4, atol=scale)
    armed = t[:, col("armed")] > 0
    assert t[armed, col("in_count")].sum() > 0
    assert t[~armed, col("in_count")].sum() == 0
    per_step = rr.split_per_control_step(t, 200)
    assert per_step.shape == (len(t)//200, len(rr.SPLIT_COLUMNS))


@pytest.mark.cpu_sim
def test_counterfactual_identity_is_bitwise_v5(baseline):
    v5bed, _, r5, s5 = baseline
    cf, _ = make_bed(CounterfactualTLMPM)
    assert cf.counterfactual.is_identity() and cf.counterfactual_label == COUNTERFACTUAL_LABEL
    r, s = run(cf)
    assert_bitwise(v5bed, cf)
    assert r5.tobytes() == r.tobytes() and s5.tobytes() == s.tobytes()


@pytest.mark.cpu_sim
def test_grid_position_update_removes_drift_from_positions(baseline):
    v5bed = baseline[0]
    cf, points = make_bed(CounterfactualTLMPM, counterfactual={"position_update": "grid_musl"})
    drift = rr.FlipDriftRecorder(cf)
    run(cf, drift)
    a = drift.arrays()
    x = cf.x.numpy().astype(np.float64)
    assert np.abs(x-points-a["ugrid_m"]).max() < 1e-6
    assert np.abs(a["drift_m"]).max() > 1e-6               # the FLIP velocity is kept (drift still accrues in v)
    assert v5bed.x.numpy().tobytes() != cf.x.numpy().tobytes()


@pytest.mark.cpu_sim
@pytest.mark.parametrize("options", [{"pic": 1.0}, {"blade_contact_stiffness": .4}, {"cutting_zone_m": .002}])
def test_each_counterfactual_option_changes_the_run(baseline, options):
    v5bed = baseline[0]
    cf, _ = make_bed(CounterfactualTLMPM, floor=False, counterfactual=options)
    ref, _ = make_bed(floor=False)
    run(cf)
    run(ref)
    assert ref.x.numpy().tobytes() != cf.x.numpy().tobytes()
    receipt = cf.counterfactual_receipt()
    assert receipt["label"] == COUNTERFACTUAL_LABEL and not receipt["identity"]
    if "pic" in options:
        assert cf.config.pic == 1.0
    if "blade_contact_stiffness" in options:
        assert cf.blade_contact_parameters()[0] == .4
    if "cutting_zone_m" in options:
        assert cf.blade_contact_parameters()[1] == .002


def test_counterfactual_options_validation():
    for bad in ({"position_update": "pic"}, {"pic": 1.5}, {"blade_contact_stiffness": 0.}, {"cutting_zone_m": -1.}):
        with pytest.raises(ValueError):
            CounterfactualOptions(**bad)
    assert CounterfactualOptions().is_identity()
    with pytest.raises(ValueError):          # the blade k is shared with the legacy tray in _rigid_contacts
        make_bed(CounterfactualTLMPM, floor=True, counterfactual={"blade_contact_stiffness": .4})


def test_counterfactual_module_is_not_reachable_from_production_code():
    src = Path(__file__).resolve().parents[1]/"src"/"robotic_cutting_sim"
    users = [p.name for p in src.glob("*.py")
             if p.name != "tlmpm_v5_counterfactual.py" and "tlmpm_v5_counterfactual" in p.read_text()]
    assert users == []
    configs = Path(__file__).resolve().parents[1]/"configs"
    named = [str(p) for p in configs.rglob("*.json") if "tlmpm_v5_counterfactual" in p.read_text()
             and "phase4_verification_protocol" not in p.name]
    assert named == []

"""CANDIDATE solver tlmpm_v6 (grid-consistent position update; WP-12 Phase 4 decision P-A, ADR 0020 proposed).

Stage S0 of ``configs/wp12/phase4_gridpos_protocol_v1.json`` on Warp's CPU backend (4 mm blocks of 270 particles,
law 8, ruler_knife_v1):

- ``position_update="flip_v5"`` is bitwise tlmpm_v5 (x, v, F, Fpi, plastic, progress, work, codes, reactions,
  stats, slot diagnostics), with the legacy tray and with a board fixture + table plane;
- ``"grid_musl_v1"`` (the default) equals the WP-12 V1-V4 COUNTERFACTUAL ``position_update='grid_musl'``
  bitwise (this file is the only place that names both modules) and makes |x - X - u_grid| < 1e-6 m;
- CPU negative controls under grid_musl_v1: side push, flat press, blunt handle and no contact give 0 labels and
  1 component; an armed stop and an armed withdrawal add no labels; a half-depth cut stays 1 component with the
  code-0 ligament below the stop;
- free block (no supports, no gravity, 2 mm lattice): per substep Delta sum(m v) equals the blade impulse to
  1e-6 of the momentum scale, for v5 and v6;
- opt-in only: no default, preset, scene or config selects tlmpm_v6; the v5 runtime paths are unchanged.
"""
from pathlib import Path
import hashlib

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from robotic_cutting_sim import root_cause_recorders as rr
from robotic_cutting_sim import tlmpm_v5, tlmpm_v6
from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, block, drive_blade
from robotic_cutting_sim.cut_topology import mechanical_components_packed
from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.packed_solvers import PACKED_SOLVERS, is_packed_solver, packed_solver_class
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

ROOT = Path(__file__).resolve().parents[1]
TOP = .012
DX = TLMPMConfig().dx
PHASES = [Phase((0, 0, -.05), .30, -1), Phase((0, 0, -.05), .26, 0), Phase((0, 0, .05), .04, -1)]
START = [.002, 0, TOP+.004]
SNAPSHOT_KEYS = ("positions", "velocities", "codes", "F", "Fpi", "plastic_strain", "cut_progress", "contact_work_j")
V5_SHA256 = "a7a05b2fe709fbff2e915c0df0855079d5bccbd8e766c787d2f628e7cc3bf34b"


def make_bed(cls=tlmpm_v6.TLMPM, floor=True, gravity=None, **kwargs):
    points, masses = block()
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    extra = {} if gravity is None else {"gravity": gravity}
    config = TLMPMConfig(cuts=1, blade_half_width=float(half), cutting_energy=5., floor=floor, fixture_walls=floor,
                         **extra)
    if floor is False and kwargs.pop("fixtures", True):
        kwargs = {"fixture_boxes": [((0., 0., -.005), (.06, .04, .005))], "table_plane": [0., 0., 1., .010], **kwargs}
    return cls(points, masses, .004, config, "cpu", blade_profile_xz=polygon, cutting_law=8, **kwargs), points


def run(bed, drift=None, phases=PHASES, start=START, rotation=VERTICAL_CUT):
    quaternion, position = rotation.as_quat(), np.asarray(start, float)
    reactions, stats = [], []
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
                bed.step()
                if drift is not None:
                    drift.after_step()
                reactions.append(bed.reaction.numpy()[0].copy())
                stats.append(bed.stats.numpy().copy())
            position = position+velocity*dt*count
    return np.asarray(reactions), np.asarray(stats)


def assert_bitwise(a, b):
    sa, sb = a.snapshot(), b.snapshot()
    for key in SNAPSHOT_KEYS:
        assert sa[key].tobytes() == sb[key].tobytes(), key
    assert a.slot_diagnostics() == b.slot_diagnostics()


@pytest.fixture(scope="module")
def v5_runs():
    out = {}
    for floor in (True, False):
        bed, points = make_bed(tlmpm_v5.TLMPM, floor=floor)
        out[floor] = (bed, points, *run(bed))
    return out


# ----------------------------------------------------------------------------- S0 bitwise and position checks

@pytest.mark.cpu_sim
@pytest.mark.parametrize("floor", [True, False])
def test_flip_v5_option_is_bitwise_v5(v5_runs, floor):
    v5bed, _, r5, s5 = v5_runs[floor]
    bed, _ = make_bed(floor=floor, position_update="flip_v5")
    assert type(bed).__module__ == "robotic_cutting_sim.tlmpm_v6" and bed.position_update == "flip_v5"
    r, s = run(bed)
    assert np.count_nonzero(v5bed.code.numpy()) > 0      # the run labels (the identity covers assignment)
    assert_bitwise(v5bed, bed)
    assert r5.tobytes() == r.tobytes() and s5.tobytes() == s.tobytes()


@pytest.mark.cpu_sim
@pytest.mark.parametrize("floor", [True, False])
def test_grid_musl_v1_equals_counterfactual_grid_positions_bitwise(floor):
    from robotic_cutting_sim.tlmpm_v5_counterfactual import CounterfactualTLMPM
    cf, _ = make_bed(CounterfactualTLMPM, floor=floor, counterfactual={"position_update": "grid_musl"})
    v6, _ = make_bed(floor=floor)
    assert v6.position_update == tlmpm_v6.DEFAULT_POSITION_UPDATE == "grid_musl_v1"
    rc, sc = run(cf)
    r6, s6 = run(v6)
    assert_bitwise(cf, v6)
    assert rc.tobytes() == r6.tobytes() and sc.tobytes() == s6.tobytes()


@pytest.mark.cpu_sim
def test_grid_musl_v1_positions_follow_the_grid_displacement(v5_runs):
    v5bed = v5_runs[True][0]
    bed, points = make_bed()
    drift = rr.FlipDriftRecorder(bed)
    run(bed, drift)
    a = drift.arrays()
    x = bed.x.numpy().astype(np.float64)
    assert np.abs(x-points.astype(np.float64)-a["ugrid_m"]).max() < 1e-6
    assert np.abs(a["drift_m"]).max() > 1e-6               # v_p (FLIP blend) is kept: drift accrues in v only
    assert v5bed.x.numpy().tobytes() != bed.x.numpy().tobytes()
    assert np.count_nonzero(bed.code.numpy()) > 0
    assert bed.slot_diagnostics()["overflow_entries"] == 0
    assert np.isfinite(bed.x.numpy()).all() and np.linalg.det(bed.F.numpy().astype(float)).min() > .5


def test_options_receipt_and_defaults():
    with pytest.raises(ValueError):
        make_bed(position_update="grid")
    bed, _ = make_bed(position_update="flip_v5")
    receipt = bed.solver_receipt()
    assert receipt["solver"] == "tlmpm_v6" and receipt["status"] == "CANDIDATE_NOT_ADOPTED"
    assert receipt["position_update"] == "flip_v5"
    assert receipt["source_sha256"]["tlmpm_v5.py"] == V5_SHA256
    assert receipt["source_sha256"]["tlmpm_v6.py"] == hashlib.sha256(Path(tlmpm_v6.__file__).read_bytes()).hexdigest()
    assert tlmpm_v6.TLMPM.code_encoding == "packed2" and issubclass(tlmpm_v6.TLMPM, tlmpm_v5.TLMPM)


def test_frozen_solver_files_unchanged():
    expected = {"tlmpm_v5.py": V5_SHA256,
                "tlmpm_v4.py": "cb8985ebbfe530be741025f009400622f6b72cdceacf239f952612c91c2eaea9",
                "tlmpm_v5_counterfactual.py": "c24f8f914781bf26063d8de886d2746305d85e8f1c37aca00ff467d244aaac43",
                "tlmpm.py": "a39c151cb8122a739b18991699527d2557219b5945d7e44be5ca50fd6e7ff7fe",
                "tlmpm_v2.py": "b2946e95dcb70492d1e06f210681788ced6e5f54fdf343aa31cd69ca85ff66b0"}
    src = ROOT/"src"/"robotic_cutting_sim"
    assert {name: hashlib.sha256((src/name).read_bytes()).hexdigest() for name in expected} == expected


# ----------------------------------------------------------------------------- S0 CPU controls (grid_musl_v1)

def probe(rotation, start, phases, **kwargs):
    return drive_blade(RULER_KNIFE_V1, rotation, start, phases, cutting_law=8, record_history=True,
                       solver_version=6, **kwargs)


def components(result):
    return mechanical_components_packed(result["reference_stencil_nodes"], result["reference_stencil_weights"],
                                        result["state"]["codes"], 1, gradients=result["reference_stencil_gradients"])[0]


@pytest.mark.cpu_sim
def test_v6_controls_side_push_flat_press_blunt_handle_no_contact_label_nothing():
    side = probe(VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)])
    flat = probe(FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)])
    handle = probe(VERTICAL_CUT, [-.05, 0, TOP+.01], [Phase((0, 0, -.05), .2)],
                   obstacle_boxes_edge=[[[0., 0., .05], [.006, .006, .006]]])
    none = probe(VERTICAL_CUT, [.001, 0, TOP+.03], [Phase((.05, 0, 0), .1), Phase((0, 0, -.05), .1)])
    for result in (side, flat, handle, none):
        assert result["solver_module"] == "robotic_cutting_sim.tlmpm_v6"
        assert result["solver_receipt"]["position_update"] == "grid_musl_v1"
        assert set(result["labelled_after_phase"]) == {0} and result["severed_bonds"] == 0
        assert components(result) == 1
    assert abs(side["forces"][0]["mean_N"][0]) > .1 and flat["forces"][0]["mean_N"][2] > 1.
    assert handle["forces"][0]["mean_N"][2] > .5


@pytest.mark.cpu_sim
def test_v6_controls_armed_stop_and_withdrawal_add_no_labels_and_half_cut_keeps_ligament():
    stop = probe(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, 0), .2)])
    withdraw = probe(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)])
    for result in (stop, withdraw):
        first, second = result["labelled_after_phase"]
        assert first > 0 and second == first
        assert result["severed_bonds_after_phase"][1] == result["severed_bonds_after_phase"][0]
        assert components(result) == 1
        assert result["cutting_state_after_phase"][1]["stroke"] == 0
    reference, codes = withdraw["points"], withdraw["state"]["codes"]
    below = reference[:, 2] < TOP-.006-1e-4
    core = np.abs(reference[:, 0]-.001) <= DX
    assert np.any(below & core) and np.all(codes[below & core] == 0)


@pytest.mark.cpu_sim
@pytest.mark.parametrize("cls", [tlmpm_v5.TLMPM, tlmpm_v6.TLMPM], ids=["v5", "v6"])
def test_free_block_momentum_change_equals_blade_impulse(cls):
    """No supports, no gravity, 2 mm lattice (off the 4 mm node planes): per substep Delta sum(m v) equals the blade
    impulse to float32 rounding (1e-6 of the total blade impulse), for v5 and v6 alike (v6 changes x, not the momentum
    update). Not used: the 4 mm test block, whose particles sit on node planes; there zero-mass corners receive the
    stress-gradient term and the grid update drops it (a frozen v4/v5 property of that lattice, not of v6; formal
    1 mm lattices sit at node fractions 1/8, 3/8, ... and have no zero-weight corners)."""
    points, masses = block(spacing=.002)
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    config = TLMPMConfig(cuts=1, blade_half_width=float(half), cutting_energy=5., floor=False, fixture_walls=False,
                         gravity=(0., 0., 0.))
    bed = cls(points, masses, .002, config, "cpu", blade_profile_xz=polygon, cutting_law=8)
    m = bed.mass.numpy().astype(np.float64)
    q, position, velocity = VERTICAL_CUT.as_quat(), np.array([-.034, 0, .003]), np.array([.05, 0, 0])
    steps, dt = 1600, bed.config.dt
    poses = np.array([np.r_[position+velocity*dt*i, q] for i in range(steps+1)], np.float32)
    bed.set_trajectory(poses, np.tile(np.r_[velocity, 0., 0., 0.], (steps+1, 1)).astype(np.float32),
                       np.zeros(steps+1, np.int32))
    errors, total = [], np.zeros(3)
    p_start = momentum = (m[:, None]*bed.v.numpy().astype(np.float64)).sum(0)
    for _ in range(steps):
        bed.reaction.zero_()
        bed.step()
        impulse = -bed.reaction.numpy()[0][:3].astype(np.float64)   # reaction = force on the blade (-j)
        after = (m[:, None]*bed.v.numpy().astype(np.float64)).sum(0)
        errors.append(np.abs(after-momentum-impulse).max())
        momentum, total = after, total+impulse
    assert np.linalg.norm(total) > 1e-4                            # the blade pushed the block
    assert max(errors) < 1e-6*np.abs(total).max()                  # per substep, scale = total blade impulse
    assert np.abs(momentum-p_start-total).max() < 1e-4*np.abs(total).max()   # accumulated float32 rounding
    assert bed.slot_diagnostics()["overflow_entries"] == 0


# ----------------------------------------------------------------------------- opt-in plumbing

def test_packed_solver_helper():
    assert PACKED_SOLVERS == ("tlmpm_v5", "tlmpm_v6")
    assert is_packed_solver("tlmpm_v5") and is_packed_solver("tlmpm_v6") and not is_packed_solver(None)
    assert not is_packed_solver("tlmpm_v4") and not is_packed_solver("tlmpm_v5_counterfactual")
    assert packed_solver_class("tlmpm_v5") is tlmpm_v5.TLMPM and packed_solver_class("tlmpm_v6") is tlmpm_v6.TLMPM
    with pytest.raises(ValueError):
        packed_solver_class("tlmpm_v4")


def test_v6_is_never_a_default():
    from robotic_cutting_sim.kinematic_env import KinematicConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from robotic_cutting_sim.scene_config import SOLVERS
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1 as profile
    assert KinematicConfig().solver is None and "tlmpm_v6" not in SOLVERS
    assert LiveCuttingMaterial(tool_profile=profile, cutting_law=8).solver is None
    material = LiveCuttingMaterial(tool_profile=profile, cutting_law=8, max_cuts=15, solver="tlmpm_v6")
    assert material.code_encoding == "packed2" and material.slots_per_node == 8
    assert KinematicConfig(max_cuts=15, solver="tlmpm_v6").max_cuts == 15
    with pytest.raises(ValueError):
        KinematicConfig(solver="tlmpm_v6", cutting_law=8, surface_profile="cut_aware_surface_v1")  # surface: v5 only
    with pytest.raises(ValueError):
        LiveCuttingMaterial(tool_profile=profile, cutting_law=7, solver="tlmpm_v6")
    # No config (scene, goal, preset, benchmark, envelope) names it, except the gridpos protocol that tests it.
    named = [str(p.relative_to(ROOT)) for p in (ROOT/"configs").rglob("*.json")
             if "tlmpm_v6" in p.read_text() and p.name != "phase4_gridpos_protocol_v1.json"]
    assert named == []
    # Only the solver itself, the helper and the opt-in plumbing name it in src/.
    src = ROOT/"src"/"robotic_cutting_sim"
    users = sorted(p.name for p in src.glob("*.py") if "tlmpm_v6" in p.read_text())
    assert users == ["blade_probe.py", "kinematic_env.py", "live_material.py", "packed_solvers.py", "tlmpm_v6.py"]


@pytest.mark.cpu_sim
def test_kinematic_env_v6_opt_in_and_v5_path_unchanged():
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    states = {}
    for solver in ("tlmpm_v5", "tlmpm_v6"):
        env = make_kinematic_env("CUT-01-R", particle_spacing_m=.004, material_dt=.0005, cutting_law=8,
                                 solver=solver, max_cuts=1)
        observation, _ = env.reset(seed=42)
        material = env.backend.material
        receipt = material.parameter_receipt
        assert type(material.bed) is (tlmpm_v5.TLMPM if solver == "tlmpm_v5" else tlmpm_v6.TLMPM)
        assert receipt["solver_module"] == f"robotic_cutting_sim.{solver}" and receipt["code_encoding"] == "packed2"
        if solver == "tlmpm_v5":
            assert "tlmpm_v6" not in receipt and not hasattr(material.bed, "position_update")
        else:
            assert receipt["tlmpm_v6"]["position_update"] == "grid_musl_v1"
            assert receipt["tlmpm_v6"]["source_sha256"]["tlmpm_v5.py"] == V5_SHA256
        pose = np.asarray(observation["proprio"]["tool_pose"], float)
        for _ in range(3):
            env.step({"mode": "cartesian_edge_pose_world", "target": pose, "cut_enabled": False})
        states[solver] = material.bed.x.numpy().copy()
        assert material.bed.check_slots()["overflow_entries"] == 0
        assert env.backend.receipt()["config"]["solver"] == solver
        env.close()
    assert np.isfinite(states["tlmpm_v6"]).all()

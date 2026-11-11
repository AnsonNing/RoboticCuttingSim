"""S1 self-penetration detector ``surface_grid_v1`` (ADR 0023 §23.2 decision 2; plan P2_REMAINING_PLAN §2.2 tests
(a)-(f)), on the Warp CPU device:

- (a) a case-C strip whose far grip is lifted, carried back over the strip and driven down through its own lower leg
  flags ``RUBBER_SELF_CONTACT_UNRESOLVED`` (8) before any other invalid bit, at the first check at or after the
  geometric overlap (an independent brute-force cKDTree search every substep), i.e. within one check interval; the
  RUBBER-PHYS-09 lambda 1.5 instantaneous-release fold scene trips the bit as well;
- (b) a prescribed legal U-bend of radius 3t never flags (float32 x and the float64 shadow); a closed U (inner gap
  below s) does;
- (c) L0: single env vs batched N = 1 (P2-6 option set, x64 positions) and N = 3 heterogeneous, byte-wise after every
  control step (solver state, detector state, invalid bits);
- (d) isolation: the crossing in env 1 changes no byte of envs 0 / 2 and sets bit 8 only for env 1;
- (e) default off_v1: no detector, no array, no launch; with the detector on the solver state is byte-identical to off;
- (f) the masked reset clears the detector state and the bit of the reset env only;
- no false trips in the v5 RUBBER-PHYS-09 cantilever / free-vibration scenes and the visco runner scenes (built by
  their unchanged builders with the detector forced on).
"""
import importlib.util
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

from robotic_cutting_sim import live_rubber_material as lrm
from robotic_cutting_sim import rubber_phys as rp
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim import rubber_self_penetration as rsp
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_material import ROOT, MaterialValues

ON = "surface_grid_v1"
CS = 8                                             # check interval of the CPU tests (substeps)
CDT = 1e-3
STEP = MaterialValues(mu0=2.5e5, kappa=5e6, density=1000.0)
P26 = {"dirichlet": "consistent_mass_projection_v1", "constraint_work": "kick_average_v1",
       "traction_work": "grid_displacement_v1", "state_precision": "f64_shadow_v1"}
# fold path of grip 1 (case C, L 24 mm): lift + carry back to x = 6 mm over 0.1 s, then down through the strip
FOLD = dict(L=0.024, h=0.009, xc=0.006, zlow=-0.006, T1=0.1, T2=0.08)
FOLD_STEPS = 140                                   # the overlap happens at control step 137 (brute force below)
SOLVER_NAMES = ("x", "v", "F", "stress", "x_start", "impulse", "gm", "gp", "gj", "old", "new_v", "jbar", "reaction",
                "work", "violations", "cfl_violations", "index")


def _same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u*u*u*(u*(6*u-15)+10)


def fold_target(t, L=FOLD["L"], h=FOLD["h"], xc=FOLD["xc"], zlow=FOLD["zlow"], T1=FOLD["T1"], T2=FOLD["T2"]):
    if t <= T1:
        u = _smooth(t/T1)
        return np.array([(xc-L)*u, 0.0, h*np.sin(np.pi/2*u)])
    u = _smooth((t-T1)/T2)
    return np.array([xc-L, 0.0, h+(zlow-h)*u])


def benign_target(t, e):
    """Small grip-1 motion (no crossing): axial stretch and a lateral sway."""
    return np.array([0.002*np.sin(np.pi*t/0.2), 0.001*(e+1)*np.sin(2*np.pi*t/0.1), 0.0])


def _live(sp=ON, L=FOLD["L"], W=0.003, mu=2e5, rho=1150.0, cs=CS, strict=False, **kw):
    spec = rs.RubberStripSpec(length=L, width=W, thickness=0.003, n_t=2, case="C", end_mask="xyz")
    return LiveRubberMaterial(spec, material=MaterialValues(mu0=mu, kappa=20*mu, density=rho), control_dt=CDT,
                              gravity=(0, 0, 0), strict=strict, step_material=STEP, self_penetration_check=sp,
                              self_penetration_check_substeps=cs, **kw)


def _solver_bytes_equal(a, b):
    for name in SOLVER_NAMES:
        assert _same(getattr(a, name).numpy(), getattr(b, name).numpy()), name


# ----------------------------------------------------------------------------- helpers / options

def test_check_positions_and_chunk_alignment():
    assert rsp.check_positions(5226, 256) == list(range(256, 5226, 256))+[5226]
    assert rsp.check_positions(512, 256) == [256, 512]
    assert rsp.check_positions(100, 256) == [100]
    assert rsp.aligned_chunk(500, 5226, 256) == 256
    assert rsp.aligned_chunk(1000, 5226, 256) == 768
    assert rsp.aligned_chunk(100, 5226, 256) == 256
    assert rsp.aligned_chunk(500, 100, 256) == 100
    assert rsp.aligned_chunk(500, 300, 256) == 300          # whole control step: one replay
    with pytest.raises(ValueError):
        rsp.check_option("hashgrid", 256)
    with pytest.raises(ValueError):
        rsp.check_option(ON, 0)


def test_invalid_reason_tables():
    for table in (lrm.INVALID_REASONS, rmb.INVALID_REASONS):
        assert table[8] == "RUBBER_SELF_CONTACT_UNRESOLVED"
        assert {1, 2, 4} <= set(table)
    assert lrm.decode_reason(8 | 2) == rmb.decode_reason(10) == ["RUBBER_J_DOMAIN", "RUBBER_SELF_CONTACT_UNRESOLVED"]


def test_no_hashgrid_in_the_detector():
    src = (ROOT/"src"/"robotic_cutting_sim"/"rubber_self_penetration.py").read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "HashGrid(" not in code and "hash_grid_query" not in code


def test_default_off_builds_nothing_and_launches_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("detector launched although off_v1")
    monkeypatch.setattr(rsp.SelfPenetrationDetector, "check", boom)
    monkeypatch.setattr(rsp.SelfPenetrationDetector, "post", boom)
    spec = rs.RubberStripSpec(length=0.012, width=0.003, thickness=0.003, n_t=2, case="C", end_mask="xyz")
    live = LiveRubberMaterial(spec, material=MaterialValues(mu0=2e5, kappa=4e6, density=1150.0), control_dt=CDT,
                              step_material=STEP)
    assert live.self_penetration_check == "off_v1" and live.self_penetration is None
    assert not hasattr(live, "_self_penetration_invalid") and "self_penetration" not in live.receipt()
    live.advance({1: {"translation": np.array([1e-4, 0, 0])}})
    assert "self_penetration" not in live.step_record(live.last_diagnostics, live.last_wrench, 0)
    batch = rmb.BatchedRubberMaterial([_live("off_v1", L=0.012), _live("off_v1", L=0.015)], device="cpu",
                                      parity_mode=True)
    assert batch.self_penetration is None and batch.self_penetration_read() is None
    assert not any(k.startswith("self_penetration") for k in batch.device_arrays())
    assert "self_penetration" not in batch.receipt()
    batch.advance(translation=np.zeros((2, 2, 3)))


def test_mixed_options_refused_and_vars_complete():
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live(ON, L=0.012), _live("off_v1", L=0.012)], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live(ON, L=0.012, cs=8), _live(ON, L=0.012, cs=16)], device="cpu",
                                  parity_mode=True)
    batch = rmb.BatchedRubberMaterial([_live(ON, L=0.012), _live(ON, L=0.015)], device="cpu", parity_mode=True)
    sp = batch.self_penetration
    owned = {k for k, v in vars(sp).items() if hasattr(v, "numpy") and k not in ("positions", "reference")}
    assert owned == set(rsp.RESET_ARRAYS+rsp.CONSTANT_ARRAYS+rsp.SCRATCH_ARRAYS)
    assert sp.positions is batch.solver.x and sp.reference is batch.solver.reference
    r = batch.receipt()["self_penetration"]
    assert r["self_penetration_check"] == ON and r["check_substeps"] == CS and "no wp.HashGrid" in r["grid"]
    assert r["distance_threshold_m"] == batch.solver.spacing and r["reference_exclusion_m"] == 3*batch.solver.dx


# ----------------------------------------------------------------------------- (a) deliberate crossing

def test_a_fold_through_own_body_flags_within_one_check_interval():
    live = _live(ON)
    ref = _live("off_v1")
    sc = live.scene
    surf = np.flatnonzero(sc.surface)
    X = sc.X[surf].astype(np.float64)
    d, excl = sc.spacing, 3*sc.dx
    rs_ = ref.solver
    overlap, first_other, hit_step = None, None, None
    for k in range(1, FOLD_STEPS+1):
        m = {"translation": fold_target(k*CDT)}
        live.advance({1: m})
        # independent reference: the same trajectory stepped directly, brute-force overlap after every substep
        rs_.begin_control_step()
        rs_.set_group_motion(1, **m)
        for j in range(1, ref.substeps+1):
            rs_.step()
            if overlap is None:
                x = rs_.x.numpy()[surf].astype(np.float64)
                pairs = cKDTree(x).query_pairs(d, output_type="ndarray")
                if len(pairs):
                    far = np.linalg.norm(X[pairs[:, 0]]-X[pairs[:, 1]], axis=1) > excl
                    close = np.linalg.norm(x[pairs[:, 0]]-x[pairs[:, 1]], axis=1) < d
                    if (far & close).any():
                        overlap = (k, j)
        rs_.diagnostics.read()                         # advance() launches the diagnostics (J-bar) after the step
        _solver_bytes_equal(live.solver, rs_)          # the detector reads only: state identical to off
        other = live.invalid_reason & ~rsp.INVALID_RUBBER_SELF_CONTACT_UNRESOLVED
        if other and first_other is None:
            first_other = k
        if live.invalid_reason & 8 and hit_step is None:
            hit_step = k
    assert overlap is not None and overlap[0] >= 100, overlap        # crossing in the descent, not a fold artefact
    sp = live.last_self_penetration
    assert hit_step is not None and sp["hits"] > 0
    g_overlap = (overlap[0]-1)*live.substeps+overlap[1]
    g_hit = (sp["first_hit_control_step"]-1)*live.substeps+sp["first_hit_substep"]
    assert 0 <= g_hit-g_overlap < CS, (overlap, sp)
    assert hit_step == sp["first_hit_control_step"]
    assert first_other is None or first_other >= hit_step, (first_other, hit_step)
    assert sp["checks"] == FOLD_STEPS*len(rsp.check_positions(live.substeps, CS))
    rec = live.step_record(live.last_diagnostics, live.last_wrench, 0)
    assert rec["self_penetration"]["hits"] == sp["hits"] and "RUBBER_SELF_CONTACT_UNRESOLVED" in rec["invalid_flags"]
    with pytest.raises(lrm.RubberInvalidState, match="RUBBER_SELF_CONTACT_UNRESOLVED"):   # strict mode raises
        strict = _live(ON, strict=True)
        for k in range(1, FOLD_STEPS+1):
            strict.advance({1: {"translation": fold_target(k*CDT)}})


def test_a_lambda_1p5_release_fold_scene_trips(monkeypatch):
    """RUBBER-PHYS-09 v5 stretch_release (lambda 1.5 over 2 control steps, release after step 3; t 3 mm, W/t 1, n_t 3,
    guard 1.0), the fold scene of ADR 0023 §18.3, built by ``rubber_phys.case_records`` with the detector on."""
    built = _force_detector(monkeypatch, cs=256)
    cfg = rp.load_config(rp.CONFIG.parent/"rubber_phys_tests_v5.json")
    r = rp.case_records("stretch_release", 0.003, 1, cfg, stable_step_guard=1.0)
    live = built[-1]
    assert "RUBBER_SELF_CONTACT_UNRESOLVED" in r["invalid_flags"], r["step_flags"]
    assert live.last_self_penetration["hits"] > 0
    assert live.last_self_penetration["first_hit_control_step"] > 3            # after the release


# ----------------------------------------------------------------------------- (b) legal bends

def _u_bend(scene, R, x0):
    """Reference strip (x along the length, z through the thickness) bent into a U of centre-line radius R starting at
    x0: lower leg, half circle, upper leg back over the lower leg."""
    X = scene.X.astype(np.float64)
    zc = float(scene.spec.thickness)/2
    out = X.copy()
    u = X[:, 0]-x0
    bend = (u > 0) & (u <= np.pi*R)
    back = u > np.pi*R
    r = R-(X[:, 2]-zc)
    th = u/R
    out[bend, 0] = x0+r[bend]*np.sin(th[bend])
    out[bend, 2] = zc+R-r[bend]*np.cos(th[bend])
    out[back, 0] = x0-(u[back]-np.pi*R)
    out[back, 2] = zc+2*R-(X[back, 2]-zc)
    return out


@pytest.mark.parametrize("options", [{}, {"state_precision": "f64_shadow_v1"}], ids=["x", "x64"])
def test_b_prescribed_u_bend_radius_3t_never_flags(options):
    t = 0.002
    spec = rs.RubberStripSpec(length=0.04, width=0.002, thickness=t, n_t=2, case="A")
    live = LiveRubberMaterial(spec, material=MaterialValues(mu0=2e5, kappa=4e6, density=1150.0), control_dt=CDT,
                              self_penetration_check=ON, self_penetration_check_substeps=CS, **options)
    sp, s = live.self_penetration, live.solver
    target = s.x64 if options else s.x
    assert sp.positions is target
    s_ = live.scene.spacing
    # particle centres sit s/2 inside the surfaces: the inner centre gap of the U is 2R - t + s (5t + s at R = 3t);
    # R = t/2 - s/4 closes it to s/2 < s (positive control)
    for R, expect in ((3*t, 0), (5*t, 0), (t/2-s_/4, None)):
        sp.reset_all()
        pos = _u_bend(live.scene, R, 0.008)
        target.assign(pos if options else pos.astype(np.float32))
        sp.check()
        hits = sp.read()[0]["hits"]
        if expect is None:
            assert hits > 0, R               # closed U: inner centre gap s/2 < s
        else:
            assert hits == 0, R


# ----------------------------------------------------------------------------- (c) L0, (d) isolation, (f) reset

def _batch_inputs(k, fold_env, N):
    tr = np.zeros((N, 2, 3))
    for e in range(N):
        tr[e, 1] = fold_target(k*CDT) if e == fold_env else benign_target(k*CDT, e)
    return tr


def test_c_l0_n1_p26_options_single_vs_batched():
    single = _live(ON, **P26)
    batch = rmb.BatchedRubberMaterial([_live(ON, **P26)], device="cpu", parity_mode=True, strict=False)
    assert batch.self_penetration.positions is batch.solver.x64
    for k in range(1, FOLD_STEPS+1):
        tr = _batch_inputs(k, 0, 1)
        single.advance({1: {"translation": tr[0, 1]}})
        batch.advance(translation=tr)
        _compare_env(batch, 0, single)
    assert int(batch.invalid_reason.numpy()[0]) & 8 and single.invalid_reason & 8


def _compare_env(batch, e, live):
    snap = batch.solver.snapshot(e)
    s = live.solver
    for name in ("x", "v", "F", "stress", "x64", "F64"):
        if name in snap:
            assert _same(snap[name], getattr(s, name).numpy()), name
    assert _same(batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]), "diagnostics"
    bsp, ssp = batch.self_penetration, live.self_penetration
    for name in rsp.RESET_ARRAYS:
        assert _same(getattr(bsp, name).numpy()[e:e+1], getattr(ssp, name).numpy()), name
    assert int(batch.invalid_reason.numpy()[e]) == live.invalid_reason
    assert batch.self_penetration_read()[e] == live.last_self_penetration


def test_c_d_f_l0_n3_isolation_and_masked_reset():
    kws = [dict(L=0.018), dict(), dict(L=0.021, W=0.0045, mu=1.5e5, rho=1200.0)]
    singles = [_live(ON, **kw) for kw in kws]
    batch = rmb.BatchedRubberMaterial([_live(ON, **kw) for kw in kws], device="cpu", parity_mode=True, strict=False)
    control = rmb.BatchedRubberMaterial([_live(ON, **kw) for kw in kws], device="cpu", parity_mode=True, strict=False)
    for k in range(1, FOLD_STEPS+1):
        tr = _batch_inputs(k, 1, 3)
        batch.advance(translation=tr)
        control.advance(translation=_batch_inputs(k, -1, 3))        # env 1 benign in the control batch
        for e in range(3):
            singles[e].advance({1: {"translation": tr[e, 1]}})
            _compare_env(batch, e, singles[e])                      # (c) L0 byte-wise, every control step
        for e in (0, 2):                                            # (d) isolation
            a, b = batch.solver.snapshot(e), control.solver.snapshot(e)
            for name in a:
                if name != "index":
                    assert _same(a[name], b[name]), (e, name)
            for name in rsp.RESET_ARRAYS:
                assert _same(getattr(batch.self_penetration, name).numpy()[e],
                             getattr(control.self_penetration, name).numpy()[e]), (e, name)
    bits = batch.invalid_reason.numpy()
    assert bits[1] == 8 and bits[0] == 0 and bits[2] == 0, bits
    assert control.invalid_reason.numpy().tolist() == [0, 0, 0]
    read = batch.self_penetration_read()
    assert read[1]["hits"] > 0 and read[0]["hits"] == read[2]["hits"] == 0
    # (f) masked reset of env 1: detector state and bit cleared; envs 0 / 2 unchanged
    keep = {e: (batch.solver.snapshot(e), [getattr(batch.self_penetration, n).numpy()[e] for n in rsp.RESET_ARRAYS])
            for e in (0, 2)}
    bits_before = batch.invalid_reason.numpy().copy()
    batch.reset([1])
    fresh = rmb.BatchedRubberMaterial([_live(ON, **kw) for kw in kws], device="cpu", parity_mode=True, strict=False)
    assert batch.invalid_reason.numpy()[1] == 0 and batch.invalid_reason.numpy()[[0, 2]].tolist() == \
        bits_before[[0, 2]].tolist()
    for name in rsp.RESET_ARRAYS:
        assert _same(getattr(batch.self_penetration, name).numpy()[1], getattr(fresh.self_penetration, name).numpy()[1])
    assert batch.self_penetration_read()[1] == {"hits": 0, "first_hit_check": None, "first_hit_control_step": None,
                                               "first_hit_substep": None, "checks": 0, "blind_steps": 0}
    for e, (snap, det) in keep.items():
        now = batch.solver.snapshot(e)
        for name in snap:
            if name != "index":
                assert _same(snap[name], now[name]), (e, name)
        for name, v in zip(rsp.RESET_ARRAYS, det):
            assert _same(getattr(batch.self_penetration, name).numpy()[e], v), (e, name)
    singles[1].reset()
    assert singles[1].invalid_reason == 0 and singles[1].self_penetration.read()[0]["hits"] == 0


# ----------------------------------------------------------------------------- no false trips (v5 / visco scenes)

def _force_detector(monkeypatch, cs):
    built = []

    class Forced(LiveRubberMaterial):
        def __init__(self, *a, **kw):
            kw.setdefault("self_penetration_check", ON)
            kw.setdefault("self_penetration_check_substeps", cs)
            super().__init__(*a, **kw)
            built.append(self)
    monkeypatch.setattr(rp, "LiveRubberMaterial", Forced)
    monkeypatch.setattr(lrm, "LiveRubberMaterial", Forced)
    return built


@pytest.mark.parametrize("scenario", ["cantilever_gravity", "free_vibration"])
def test_no_false_trip_v5_09_scenes(monkeypatch, scenario):
    built = _force_detector(monkeypatch, cs=256)
    cfg = rp.load_config(rp.CONFIG.parent/"rubber_phys_tests_v5.json")
    r = rp.case_records(scenario, 0.003, 1, cfg, stable_step_guard=1.0)
    live = built[-1]
    assert live.self_penetration is not None and live.last_self_penetration["checks"] > 0
    assert r["invalid_flags"] == [] and live.last_self_penetration["hits"] == 0


def test_no_false_trip_visco_runner_scenes(monkeypatch):
    path = ROOT/"scripts"/"rubber"/"p2_3"/"run_visc_v1.py"
    if not path.exists():
        pytest.skip(f"NOT_RUN: {path} absent")
    built = _force_detector(monkeypatch, cs=16)
    spec = importlib.util.spec_from_file_location("run_visc_v1_s1", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.pop("run_visc_v1_s1", None)
    spec.loader.exec_module(mod)
    cfg, _ = mod.load_protocol()
    jobs = {name: (kind, p) for name, kind, p in mod.job_list(cfg, smoke=True)}
    for name in ("v03_strip_16hz", "v02_creep", "v04b_cantilever_x1", "v01_P2_simple_shear"):
        kind, p = jobs[name]
        out = getattr(mod, f"job_{kind}")(p, True)
        live = built[-1]
        assert live.self_penetration is not None and live.last_self_penetration["checks"] > 0, name
        assert live.invalid_reason == 0 and live.last_self_penetration["hits"] == 0, (name, out["info"])

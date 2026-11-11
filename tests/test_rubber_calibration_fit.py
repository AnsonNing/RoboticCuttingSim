"""P5.3 / CAL-01 (fit part): parameter space and transforms vs the step rule, the objective's refusals (reward term,
held-out experiments, sealed paths, v1 manifests), TRF recovery with the speculative population Jacobian, the in-repo
CMA-ES, the ask / tell hook, the fit receipt, the one-shot held-out guard and the path guard. A cheap analytic
forward model replaces the simulator here (the simulator forward is covered by CAL-02 and the repeated-call test)."""
import copy
import json
import math
import os

import numpy as np
import pytest

from robotic_cutting_sim import rubber_calibration_experiments as rce
from robotic_cutting_sim import rubber_calibration_fit as rcf
from robotic_cutting_sim import rubber_calibration_synthetic as rcs
from robotic_cutting_sim import rubber_calibration_v2 as rc2

BOUNDS = {"mu_eq_pa": (1.5e5, 4.0e5), "g1": (0.02, 0.4), "tau1_s": (2e-3, 3e-2), "tau2_s": (3e-2, 0.5),
          "mu_table": (0.3, 0.8)}
TRUTH = {"mu_eq_pa": 2.6e5, "g1": 0.15, "tau1_s": 0.008, "g2": 0.1, "tau2_s": 0.12, "mu_table": 0.5}
SENSORS = {"load_cell": {"kind": "load_cell", "noise_sigma": 0.15, "resolution": 1e-3, "sample_rate_hz": 1000.0,
                         "full_scale": 50.0, "unit": "N"},
           "crosshead": {"kind": "crosshead", "noise_sigma": 1e-6, "resolution": 1e-6, "sample_rate_hz": 1000.0,
                         "full_scale": 0.1, "unit": "m"},
           "video": {"kind": "video_markers", "noise_sigma": 1e-4, "resolution": 0.0, "sample_rate_hz": 240.0,
                     "full_scale": 0.0, "unit": "m"},
           "force_plate": {"kind": "force_plate", "noise_sigma": 3e-4, "resolution": 1e-3, "sample_rate_hz": 1000.0,
                           "full_scale": 0.1, "unit": "N"}}
PROTOCOL = {"sensors": SENSORS, "measurement": {"dimension_sigma_m": 2e-5, "mass_sigma_kg": 5e-7},
            "specimens": {"ST": {"nominal_dimensions_m": {"length": 0.06, "width": 0.012, "thickness": 0.004}},
                          "SF": {"nominal_dimensions_m": {"length": 0.03, "width": 0.016, "thickness": 0.004}}},
            "experiments": [
                {"id": "T1", "type": "tensile_load_unload", "split": "fit", "specimen_id": "ST",
                 "loading": {"strain_rate_1_s": 8.0, "max_stretch": 1.5, "min_stretch": 1.1}},
                {"id": "R1", "type": "stress_relaxation", "split": "fit", "specimen_id": "ST",
                 "loading": {"ramp_s": 0.02, "hold_stretch": 1.3, "hold_s": 0.1}},
                {"id": "F1", "type": "friction_sliding", "split": "fit", "specimen_id": "SF",
                 "loading": {"push_start_s": 0.02, "duration_s": 0.06}},
                {"id": "T3", "type": "tensile_load_unload", "split": "heldout", "specimen_id": "ST",
                 "loading": {"strain_rate_1_s": 20.0, "max_stretch": 1.5, "min_stretch": 1.1}}]}
SETTINGS = {"M3_end_samples": 10, "rebound_smoothing_frames": 5, "slip_onset_threshold_m": 5e-4}
DTC = 1e-3


def space():
    return rcf.ParameterSpace(BOUNDS, g2_fraction=(0.02, 0.98), g_total_max=0.5)


def toy_channels(theta, exp):
    """Analytic stand-in for the simulator: 1D Prony uniaxial force for T / R, a friction-limited slide for F."""
    K = int(math.ceil(rce.duration_s(exp)/DTC-1e-9)) if exp["type"] != "friction_sliding" else 60
    t = DTC*np.arange(K+1)
    if exp["type"] == "friction_sliding":
        mu = theta["mu_table"]
        x = np.where(t < 0.02, 0.0, (t-0.02)**2*(1.0-mu)*4.0)
        tv = rce.video_times(t[-1])
        fx = mu*0.0217*np.minimum(1.0, t/0.02)
        return {"force": {"time_s": t[1:], "friction_force_n": fx[1:], "normal_force_n": np.full(K, 0.0217)},
                "video": {"time_s": tv, "centroid_x_m": np.interp(tv, t, x)}, "invalid_reason": 0,
                "oracle": {"volume_ratio": [1.0]*(K+1), "dissipated_energy_j": [0.0, 1e-5*mu]}}
    lam = 1.0+np.array([rce.crosshead(exp, x) for x in t])/exp["specimen"]["length"]
    s = lam-lam**-2.0
    g1, g2 = theta["g1"], theta["g2"]
    mu0 = theta["mu_eq_pa"]/(1-g1-g2)
    h1 = h2 = 0.0
    P = [0.0]
    for k in range(1, K+1):
        ds = s[k]-s[k-1]
        h1 = h1*math.exp(-DTC/theta["tau1_s"])+g1*ds*math.exp(-DTC/(2*theta["tau1_s"]))
        h2 = h2*math.exp(-DTC/theta["tau2_s"])+g2*ds*math.exp(-DTC/(2*theta["tau2_s"]))
        P.append(mu0*((1-g1-g2)*s[k]+h1+h2))
    F = np.array(P)*0.012*0.004
    diss = float(np.sum(0.5*(F[1:]+F[:-1])*np.diff(lam*0.06)))
    return {"force": {"time_s": t[1:], "displacement_m": (lam[1:]-1)*0.06, "force_n": F[1:]}, "invalid_reason": 0,
            "oracle": {"volume_ratio": [1.0]*(K+1), "dissipated_energy_j": [0.0, diss]}}


class ToyForward:
    def __init__(self):
        self.calls = []

    def __call__(self, thetas, experiments):
        self.calls.append(len(thetas))
        return [{e["id"]: toy_channels(th, e) for e in experiments} for th in thetas]


def make_dataset(tmp_path, noise_seed=11, freeze=True, name="ds"):
    exps = rcs.protocol_experiments(PROTOCOL)
    clean = {e["id"]: toy_channels(TRUTH, e) for e in exps}
    rng = np.random.default_rng(noise_seed)
    noisy = rcs.noisy_channels(PROTOCOL, exps, clean, rng)
    specimens = rcs.measured_specimens(PROTOCOL, 1150.0, rng)
    oracle = {e["id"]: clean[e["id"]]["oracle"] for e in exps if e["split"] == "heldout"}
    syn = {"generator_commit": "0"*40, "generator_tree_clean": True, "truth_profile": {"n_t": 6, "cfl": 0.25},
           "sensor_model_id": "test", "sealed_truth_sha256": "b"*64, "prior_box": {}, "noise_seed_sha256": "c"*64}
    d = tmp_path/name
    path = rcs.write_dataset(d, protocol=PROTOCOL, protocol_sha256="a"*64, experiments=exps, noisy=noisy,
                             specimens=specimens, oracle=oracle, dataset_id=name, status="SYNTHETIC_PILOT",
                             synthetic=syn, note="toy test dataset")
    if freeze:
        m = rc2.freeze(json.loads(path.read_text()), d, "2026-11-11T00:00:00Z")
        path.unlink()
        rc2.write_manifest(path, m)
    return path


# ---------------------------------------------------------------------------- parameter space

def test_parameter_space_roundtrip_bounds_and_step_rule():
    sp = space()
    u = sp.to_u(TRUTH)
    back = sp.from_u(u)
    assert all(math.isclose(back[k], TRUTH[k], rel_tol=1e-12) for k in rcf.PARAMS)
    assert sp.inside(u)
    pts = sp.latin_hypercube(64, seed=3)
    assert np.array_equal(pts, sp.latin_hypercube(64, seed=3))
    for p in pts:
        th = sp.from_u(p)
        assert sp.inside(p) and th["g1"]+th["g2"] < 0.5
        mat = rce.material_from_theta(th, 1000.0)
        assert mat.mu0 <= sp.mu0_max()*(1+1e-12) and math.isclose(sp.mu0_max(), 8e5)
    with pytest.raises(ValueError):
        rcf.ParameterSpace({**BOUNDS, "g1": (0.02, 0.6)}, g_total_max=0.5)
    J = sp.jacobian_theta_u(u)
    assert math.isclose(J[0, 0], TRUTH["mu_eq_pa"], rel_tol=1e-6)


def test_step_rule_covers_the_box():
    """The fit configuration's step material (mu0_max = mu_eq_max / (1 - G), rho_min) bounds every box candidate."""
    sp = space()
    cfg = rce.ModelConfig()
    assert math.isclose(cfg.mu0_max, sp.mu0_max())
    worst = rce.material_from_theta(sp.from_u(sp.ub), cfg.density_min)
    rce.check_step_rule(cfg, [worst], 0.001)


# ---------------------------------------------------------------------------- objective refusals

def test_objective_refusals(tmp_path):
    path = make_dataset(tmp_path)
    with pytest.raises(rcf.CalibrationFitError, match="REWARD_TERM_REFUSED"):
        rcf.Objective(path, forward=ToyForward(), reward=lambda *a: 0.0)
    with pytest.raises(rcf.CalibrationFitError, match="REWARD_TERM_REFUSED"):
        rcf.Objective(path, forward=ToyForward(), rl_reward_weight=1.0)
    with pytest.raises(rcf.CalibrationFitError, match="HELDOUT_EXPERIMENT_REFUSED"):
        rcf.Objective(path, forward=ToyForward(), experiments=["T1", "T3"])
    sealed = tmp_path/"sealed"
    sealed.mkdir()
    os.symlink(path.parent, sealed/"ds")
    with pytest.raises(rcf.CalibrationFitError, match="SEALED_PATH_REFUSED"):
        rcf.Objective(sealed/"ds"/"manifest.json", forward=ToyForward())
    (sealed/"truth.json").write_text("{}")
    link = tmp_path/"innocent"
    os.symlink(sealed, link)
    with pytest.raises(rcf.CalibrationFitError, match="SEALED_PATH_REFUSED"):
        rcf.guard_path(link/"truth.json")                    # resolves into sealed/
    v1 = tmp_path/"v1.json"
    v1.write_text((rc2.GATE_DIR/"none_v1.json").read_text())
    with pytest.raises(rc2.CalibrationManifestV2Error):
        rcf.Objective(v1, forward=ToyForward())
    obj = rcf.Objective(path, forward=ToyForward())
    assert obj.exp_ids == ["T1", "R1", "F1"] and "T3" not in obj.data


# ---------------------------------------------------------------------------- methods

def test_trf_recovers_toy_truth_with_one_population_per_evaluation(tmp_path):
    path = make_dataset(tmp_path)
    fwd = ToyForward()
    obj = rcf.Objective(path, forward=fwd)
    sp = space()
    starts = sp.latin_hypercube(2, seed=7)
    res = rcf.fit_trf(obj, sp, starts, max_nfev=40, xtol=1e-8, fd_step=1e-4)
    best = res["starts"][res["best"]]
    assert set(fwd.calls) == {sp.p+1}               # every forward call is one population of p + 1 candidates
    assert obj.evaluations == sum(len(s["history"]) for s in res["starts"])
    th = best["theta"]
    cov, sigma, s2 = rcf.covariance(sp, best["u"], best["jac"], best["residuals"])
    for k in rcf.PARAMS:
        assert abs(th[k]-TRUTH[k]) <= max(4*sigma[rcf.PARAMS.index(k)], 0.02*TRUTH[k]), (k, th[k], TRUTH[k])
    assert 0.5 < best["chi2_reduced"] < 2.0


def test_negative_control_subset_leaves_friction_unidentifiable(tmp_path):
    path = make_dataset(tmp_path)
    obj = rcf.Objective(path, forward=ToyForward(), experiments=["T1"])
    sp = space()
    res = rcf.fit_trf(obj, sp, sp.latin_hypercube(1, seed=1), max_nfev=15)
    best = res["starts"][0]
    _, sigma, _ = rcf.covariance(sp, best["u"], best["jac"], best["residuals"])
    i = rcf.PARAMS.index("mu_table")
    assert not sigma[i]/best["theta"]["mu_table"] < 0.5         # reported non-identifiable


def test_cmaes_converges_and_is_seeded():
    target = np.array([0.3, -0.2, 0.1, 0.5, -0.4, 0.2])

    def run(seed):
        es = rcf.CMAES(np.zeros(6), 0.5, 8, seed)
        for _ in range(120):
            X = es.ask()
            es.tell([float(np.sum((x-target)**2)) for x in X])
        return es
    a, b = run(4), run(4)
    assert np.array_equal(a.mean, b.mean)
    assert np.linalg.norm(a.mean-target) < 1e-4


def test_ask_tell_hook(tmp_path):
    path = make_dataset(tmp_path)
    obj = rcf.Objective(path, forward=ToyForward())
    sp = space()
    u0 = sp.latin_hypercube(1, seed=2)[0]
    es = rcf.CMAES(u0, 0.3, 7, seed=5, lb=sp.lb, ub=sp.ub)
    first = 0.5*float(obj.residuals_population([sp.from_u(u0)])[0] @ obj.residuals_population([sp.from_u(u0)])[0])
    out = rcf.run_ask_tell(es, obj, sp, 15)
    assert out["best_loss"] < first and len(out["history"]) == 15

    class External:            # what the external agentic optimizer implements
        def __init__(self):
            self.told = []

        def ask(self):
            return np.tile(u0, (3, 1))

        def tell(self, losses):
            self.told.append(list(losses))
    ext = External()
    rcf.run_ask_tell(ext, obj, sp, 2)
    assert len(ext.told) == 2 and len(ext.told[0]) == 3

    class Bad(External):
        def ask(self):
            return np.zeros((3, 4))
    with pytest.raises(rcf.CalibrationFitError, match="ASK_SHAPE"):
        rcf.run_ask_tell(Bad(), obj, sp, 1)


# ---------------------------------------------------------------------------- freeze and one-shot held-out

def _fit_receipt(tmp_path, path, name="fit.json", experiments=None):
    obj = rcf.Objective(path, forward=ToyForward(), experiments=experiments)
    sp = space()
    res = rcf.fit_trf(obj, sp, sp.latin_hypercube(1, seed=7), max_nfev=40, xtol=1e-8, fd_step=1e-4)
    rec, sha = rcf.freeze_fit(tmp_path/name, objective=obj, space=sp, trf=res, provenance={"commit": "test"})
    return tmp_path/name, rec, sha


def _predict(theta, exps):
    return {e["id"]: toy_channels(theta, e) for e in exps}


def test_freeze_fit_receipt(tmp_path):
    path = make_dataset(tmp_path)
    p, rec, sha = _fit_receipt(tmp_path, path)
    assert rec["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED"
    assert rec["validation_kind"] == "NUMERICAL_VALIDATION_SYNTHETIC" and rec["recipe_promoted"] is False
    assert rec["objective"]["fit_experiments"] == ["T1", "R1", "F1"] and len(sha) == 64
    with pytest.raises(rcf.CalibrationFitError, match="RECEIPT_EXISTS"):
        rcf.freeze_fit(p, objective=rcf.Objective(path, forward=ToyForward()), space=space(),
                       trf={"starts": [], "best": 0}, provenance={})


def test_heldout_is_one_shot_and_guarded(tmp_path):
    path = make_dataset(tmp_path)
    fit, _, _ = _fit_receipt(tmp_path, path)
    neg, _, _ = _fit_receipt(tmp_path, path, "neg.json", experiments=["T1"])
    thr = {"T3": {"M1": 0.05, "M2": 0.05, "M8": 1e-3, "M9": 0.05}}
    ledger = tmp_path/"ledger"
    out = rcf.evaluate_heldout(path, [fit, neg], ledger, predict=_predict, thresholds=thr, metric_settings=SETTINGS)
    assert out["label"] == "ONE_SHOT" and out["verdict"] in ("PASS", "FAIL") and len(out["fits"]) == 2
    assert {r["metric"] for r in out["fits"][0]["rows"]} == {"M1", "M2", "M8", "M9"}
    with pytest.raises(rcf.CalibrationFitError, match="HELDOUT_ALREADY_EVALUATED"):
        rcf.evaluate_heldout(path, [fit], ledger, predict=_predict, thresholds=thr, metric_settings=SETTINGS)
    audit = rcf.evaluate_heldout(path, [fit], ledger, predict=_predict, thresholds=thr, metric_settings=SETTINGS,
                                 audit_rerun="reviewer_check_1")
    assert audit["label"].startswith("AUDIT_RERUN")
    with pytest.raises(rcf.CalibrationFitError, match="AUDIT_NAME_USED"):
        rcf.evaluate_heldout(path, [fit], ledger, predict=_predict, thresholds=thr, metric_settings=SETTINGS,
                             audit_rerun="reviewer_check_1")
    with pytest.raises(rcf.CalibrationFitError, match="SEALED_PATH_REFUSED"):
        rcf.evaluate_heldout(path, [fit], tmp_path/"sealed", predict=_predict, thresholds=thr,
                             metric_settings=SETTINGS)


def test_heldout_refusals(tmp_path):
    pilot = make_dataset(tmp_path, freeze=False, name="pilot")
    frozen = make_dataset(tmp_path, name="frozen")
    other = make_dataset(tmp_path, noise_seed=12, name="other")
    fit, rec, _ = _fit_receipt(tmp_path, frozen)
    thr = {"T3": {"M1": 1.0, "M2": 1.0, "M8": 1.0, "M9": 1.0}}
    kw = dict(predict=_predict, thresholds=thr, metric_settings=SETTINGS)
    with pytest.raises(rcf.CalibrationFitError, match="MANIFEST_NOT_FROZEN"):
        rcf.evaluate_heldout(pilot, [fit], tmp_path/"l1", **kw)
    with pytest.raises(rcf.CalibrationFitError, match="FIT_RECEIPT_OTHER_FREEZE"):
        rcf.evaluate_heldout(other, [fit], tmp_path/"l2", **kw)
    bad = copy.deepcopy(rec)
    bad["objective"]["fit_experiments"] = ["T1", "T3"]
    (tmp_path/"bad.json").write_text(json.dumps(bad))
    with pytest.raises(rcf.CalibrationFitError, match="FIT_USED_HELDOUT"):
        rcf.evaluate_heldout(frozen, [tmp_path/"bad.json"], tmp_path/"l3", **kw)
    assert not (tmp_path/"l3").exists() or not any((tmp_path/"l3").iterdir())     # a refusal writes no ledger


def test_reveal_checks_the_sealed_sha(tmp_path):
    p, sha = rcs.seal(tmp_path/"sealed", {"theta": TRUTH})
    assert rcs.reveal(p, sha)["theta"] == TRUTH
    with pytest.raises(ValueError, match="mismatch"):
        rcs.reveal(p, "0"*64)
    with pytest.raises(FileExistsError):
        rcs.seal(tmp_path/"sealed", {"theta": TRUTH})
    with pytest.raises(ValueError, match="sealed"):
        rcs.seal(tmp_path/"open", {})
    a, b = rcs.secret_seed(), rcs.secret_seed()
    assert a != b and len(rcs.seed_sha256(a)) == 64
    th, rho = rcs.draw_truth({**{k: v for k, v in BOUNDS.items()}, "g2": (0.06, 0.15),
                              "density_kg_m3": (1100.0, 1200.0)}, 123)
    assert th == rcs.draw_truth({**BOUNDS, "g2": (0.06, 0.15), "density_kg_m3": (1100.0, 1200.0)}, 123)[0]
    assert 1100 <= rho <= 1200 and all(BOUNDS[k][0] <= th[k] <= BOUNDS[k][1] for k in BOUNDS)


def test_lockstep_starts_equal_sequential_starts(tmp_path):
    """Lockstep (one forward population per round over every active start) gives the per-start results of the
    sequential fit, bitwise, with fewer forward calls."""
    path = make_dataset(tmp_path)
    sp = space()
    starts = sp.latin_hypercube(3, seed=9)
    seq_fwd, lock_fwd = ToyForward(), ToyForward()
    seq = rcf.fit_trf(rcf.Objective(path, forward=seq_fwd), sp, starts, max_nfev=12, fd_step=1e-4)
    lock = rcf.fit_trf_lockstep(rcf.Objective(path, forward=lock_fwd), sp, starts, max_nfev=12, fd_step=1e-4)
    for a, b in zip(seq["starts"], lock["starts"]):
        assert a["u"] == b["u"] and a["cost"] == b["cost"] and a["nfev"] == b["nfev"]
        assert np.array_equal(a["jac"], b["jac"])
    assert lock["best"] == seq["best"] and len(lock_fwd.calls) < len(seq_fwd.calls)
    assert max(lock_fwd.calls) == 3*(sp.p+1)


def test_material_dedup_shares_envs():
    """mu_table has no effect without contact: candidates differing only in mu_table share one env."""
    a = rce.material_from_theta(TRUTH, 1150.0)
    b = rce.material_from_theta(dict(TRUTH, mu_table=0.7), 1150.0)
    tens = {"id": "T", "type": "tensile_load_unload"}
    fric = {"id": "F", "type": "friction_sliding"}
    assert rce.material_key(tens, a) == rce.material_key(tens, b)
    assert rce.material_key(fric, a) != rce.material_key(fric, b)


# ---------------------------------------------------------------------------- v2 staged fit with a budget

TOY_STAGES = (
    {"name": "S1_elastic", "params": ["mu_eq_pa"], "experiments": ["T1"], "starts": 1, "max_nfev": 8},
    {"name": "S2_visco", "params": ["g1", "tau1_s", "g2", "tau2_s"], "experiments": ["R1", "T1"], "starts": 2,
     "max_nfev": 25},
    {"name": "S3_friction", "params": ["mu_table"], "experiments": ["F1"], "starts": 1, "max_nfev": 8},
    {"name": "S4_joint_polish", "params": list(rcf.PARAMS), "experiments": None, "starts": 1, "max_nfev": 10},
)
PRIOR = {"mu_eq_pa": (2e5, 3.5e5), "g1": (0.08, 0.2), "g2": (0.06, 0.15), "tau1_s": (5e-3, 1.5e-2),
         "tau2_s": (6e-2, 0.2), "mu_table": (0.4, 0.6)}


def _box(sp):
    lg = lambda p: math.log(p/(1-p))   # noqa: E731
    G = sp.G
    lo = {"mu_eq_pa": math.log(PRIOR["mu_eq_pa"][0]), "g1": lg(PRIOR["g1"][0]/G),
          "g2": lg(PRIOR["g2"][0]/(G-PRIOR["g1"][0])), "tau1_s": math.log(PRIOR["tau1_s"][0]),
          "tau2_s": math.log(PRIOR["tau2_s"][0]), "mu_table": PRIOR["mu_table"][0]}
    hi = {"mu_eq_pa": math.log(PRIOR["mu_eq_pa"][1]), "g1": lg(PRIOR["g1"][1]/G),
          "g2": lg(PRIOR["g2"][1]/(G-PRIOR["g1"][1])), "tau1_s": math.log(PRIOR["tau1_s"][1]),
          "tau2_s": math.log(PRIOR["tau2_s"][1]), "mu_table": PRIOR["mu_table"][1]}
    return np.array([lo[k] for k in rcf.PARAMS]), np.array([hi[k] for k in rcf.PARAMS])


def _centre():
    return {"mu_eq_pa": math.sqrt(2e5*3.5e5), "g1": 0.14, "g2": 0.105, "tau1_s": math.sqrt(5e-3*1.5e-2),
            "tau2_s": math.sqrt(6e-2*0.2), "mu_table": 0.5}


def test_subspace_maps_to_full_theta():
    sp = space()
    sub = rcf.SubSpace(sp, ["g1", "tau2_s"], TRUTH)
    th = sub.from_u(sub.to_u(TRUTH))
    assert all(math.isclose(th[k], TRUTH[k], rel_tol=1e-12) for k in rcf.PARAMS)
    th2 = sub.from_u(sub.to_u(TRUTH)+np.array([0.1, 0.0]))
    assert math.isclose(th2["mu_eq_pa"], TRUTH["mu_eq_pa"], rel_tol=1e-12) and th2["g1"] != TRUTH["g1"]
    assert math.isclose(th2["tau2_s"], TRUTH["tau2_s"], rel_tol=1e-12)


def test_staged_fit_recovers_toy_truth(tmp_path):
    path = make_dataset(tmp_path)
    sp = space()
    calls = []

    def make(exps):
        calls.append(exps)
        return rcf.Objective(path, forward=ToyForward(), experiments=exps)
    res = rcf.fit_staged(make, sp, _box(sp), stages=TOY_STAGES, start_theta=_centre(), budget_s=600, fd_step=1e-4)
    assert [s["name"] for s in res["stages"]] == [s["name"] for s in TOY_STAGES] and not res["budget_hit"]
    assert calls == [["T1"], ["R1", "T1"], ["F1"], None]
    _, sigma, _ = rcf.covariance(sp, res["u"], res["jac"], res["residuals"])
    for i, k in enumerate(rcf.PARAMS):
        assert abs(res["theta"][k]-TRUTH[k]) <= max(4*sigma[i], 0.02*TRUTH[k]), (k, res["theta"][k], TRUTH[k])
    lo, hi = _box(sp)
    assert np.all(np.array(res["u"]) >= lo-1e-12) and np.all(np.array(res["u"]) <= hi+1e-12)
    rec, sha = rcf.freeze_fit_staged(tmp_path/"fit_v2.json", result=res, space=sp, provenance={"test": True},
                                     protocol_id="rubber_calib_tests_v2")
    assert rec["stages"][0]["name"] == "S1_elastic" and len(sha) == 64 and rec["budget_hit"] is False


def test_budget_rule_keeps_best_iterate_and_one_final_evaluation(tmp_path):
    """Budget exhausted: the remaining stages are skipped, the final stage still gets its first evaluation, and the
    best iterate is the result (scored normally, never NOT_RUN)."""
    path = make_dataset(tmp_path)
    sp = space()
    fwd = ToyForward()
    res = rcf.fit_staged(lambda e: rcf.Objective(path, forward=fwd, experiments=e), sp, _box(sp),
                         stages=TOY_STAGES, start_theta=_centre(), budget_s=0.0)
    assert res["budget_hit"] and [s.get("skipped") for s in res["stages"][:3]] == ["BUDGET_EXHAUSTED"]*3
    assert fwd.calls == [sp.p+1]                        # exactly one population of the final stage
    assert all(math.isclose(res["theta"][k], _centre()[k], rel_tol=1e-9) for k in rcf.PARAMS)
    assert res["jac"].shape == (res["n_residuals"], sp.p)


def test_budget_counts_only_the_supplied_clock(tmp_path):
    """v2: the budget clock may be the GPU-lock hold time; time outside the clock is not counted."""
    path = make_dataset(tmp_path)
    sp = space()
    held = {"t": 0.0}

    class Clocked:
        def __init__(self, obj):
            self.obj = obj

        def __getattr__(self, n):
            return getattr(self.obj, n)

        def residuals_population(self, thetas):
            held["t"] += 10.0                          # every evaluation "holds the GPU" for 10 s
            return self.obj.residuals_population(thetas)
    res = rcf.fit_staged(lambda e: Clocked(rcf.Objective(path, forward=ToyForward(), experiments=e)), sp, _box(sp),
                         stages=TOY_STAGES, start_theta=_centre(), budget_s=55.0, clock=lambda: held["t"])
    # exceeded by at most the guaranteed final-stage population; time outside the clock is not counted
    assert res["budget_hit"] and held["t"] <= 55.0+10.0 and res["budget_clock_used_s"] == held["t"]

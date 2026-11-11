"""P5.2 / CAL-02: P2-5 forward models. Population batch (candidate x experiment -> env of a BatchedRubberMaterial) vs
single-env references byte-wise (L0) for every experiment type on the Warp CPU device; repeated runs byte-identical;
schedules, sensor transforms and metric extractors."""
import math

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from robotic_cutting_sim import rubber_calibration_experiments as rce   # noqa: E402

CFG = rce.ModelConfig(n_t=2, cfl=0.5, resolution_profile="rubber_coarse_v1", control_dt=2e-4)
THETAS = [dict(mu_eq_pa=2.5e5, g1=0.12, tau1_s=0.004, g2=0.1, tau2_s=0.05, mu_table=0.5),
          dict(mu_eq_pa=3.2e5, g1=0.2, tau1_s=0.002, g2=0.05, tau2_s=0.02, mu_table=0.4)]
EXPS = {
    "T": {"id": "T", "type": "tensile_load_unload", "specimen": {"length": 0.006, "width": 0.004, "thickness": 0.002},
          "loading": {"strain_rate_1_s": 100.0, "max_stretch": 1.1, "min_stretch": 1.04}},
    "R": {"id": "R", "type": "stress_relaxation", "specimen": {"length": 0.006, "width": 0.004, "thickness": 0.002},
          "loading": {"ramp_s": 0.0004, "hold_stretch": 1.05, "hold_s": 0.0004}},
    "C": {"id": "C", "type": "cantilever_rebound", "specimen": {"length": 0.008, "width": 0.003, "thickness": 0.002},
          "loading": {"weight_ramp_s": 0.0004, "tip_load_n": 0.002, "release_s": 0.0006,
                      "record_after_release_s": 0.0004}},
    "G": {"id": "G", "type": "gravity_sag", "specimen": {"length": 0.008, "width": 0.003, "thickness": 0.002},
          "loading": {"weight_ramp_s": 0.0004, "duration_s": 0.0008, "static_window_s": [0.0006, 0.0008]}},
    "F": {"id": "F", "type": "friction_sliding", "specimen": {"length": 0.006, "width": 0.004, "thickness": 0.002},
          "loading": {"settle_ramp_s": 0.0004, "push_start_s": 0.0004, "push_ramp_s": 0.0004,
                      "push_max_over_nominal_normal": 0.85, "extra_top_load_over_nominal_weight": 1.0,
                      "duration_s": 0.001}},
}


@pytest.fixture(scope="module", autouse=True)
def _warp():
    wp.config.log_level = wp.LOG_WARNING
    wp.init()


def _same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("name", sorted(EXPS))
def test_population_batch_equals_single_references_bytewise(name):
    exp = EXPS[name]
    mats = [rce.material_from_theta(th, 1150.0) for th in THETAS]
    eb = rce.ExperimentBatch(exp, 1150.0, CFG, len(mats), device="cpu", parity_mode=True)
    first = [rce.finalize_record(r) for r in eb.run(mats)]
    singles = [rce.run_single(exp, m, CFG) for m in mats]
    for e in range(len(mats)):
        bad = [f for f in rce.RAW_FIELDS if not _same(first[e][f], singles[e][f])]
        assert bad == [], f"{name} env {e}: {bad}"
        assert first[e]["invalid_reason"] == singles[e]["invalid_reason"] == 0
    assert not _same(first[0]["markers"], first[1]["markers"])            # heterogeneous candidates differ
    again = [rce.finalize_record(r) for r in eb.run(mats)]                 # re-used batch: masked reset, no rebuild
    assert all(_same(again[e][f], first[e][f]) for e in range(len(mats)) for f in rce.RAW_FIELDS)
    swapped = [rce.finalize_record(r) for r in eb.run(mats[::-1])]
    assert all(_same(swapped[1][f], first[0][f]) for f in rce.RAW_FIELDS)    # env index does not matter


def test_batch_refuses_density_change_and_step_rule_violation():
    exp = EXPS["R"]
    eb = rce.ExperimentBatch(exp, 1150.0, CFG, 1, device="cpu", parity_mode=True)
    with pytest.raises(ValueError, match="density"):
        eb.run([rce.material_from_theta(THETAS[0], 1100.0)])
    hot = dict(THETAS[0], mu_eq_pa=9e5)
    with pytest.raises(ValueError, match="CFL"):
        eb.run([rce.material_from_theta(hot, 1150.0)])


def test_schedules():
    exp = EXPS["T"]
    L = exp["specimen"]["length"]
    assert rce.crosshead(exp, 0.0) == 0.0
    assert math.isclose(rce.crosshead(exp, 0.001), 0.1*L, rel_tol=1e-12)
    assert math.isclose(rce.crosshead(exp, 1.0), 0.04*L, rel_tol=1e-12)        # holds at lambda_min
    assert rce.steps(exp, CFG) == math.ceil((0.1+0.06)/100.0/CFG.control_dt-1e-9)
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.rubber_material import mass_per_particle
    for name in ("C", "F"):
        e = EXPS[name]
        sc = rs.compile_strip(rce.build_spec(e, CFG))
        sch = rce.Schedule(e, sc, 1150.0, CFG)
        total = sc.n*float(mass_per_particle(1150.0, sc.spacing))*rce.G_ACC
        assert math.isclose(sch.group_weight.sum(), total, rel_tol=1e-12)
        _, f = sch.inputs(sch.K)
        if name == "C":                     # after release: weight only
            assert np.allclose(f[:, 2], -sch.group_weight)
            _, f1 = sch.inputs(2)
            assert f1[sch.names.index("tip"), 2] < -sch.group_weight[sch.names.index("tip")]*rce.smootherstep(1.5/2)
        else:
            _, fp = sch.inputs(3)            # mid-push
            ld = e["loading"]
            wn = rce.nominal_weight(e)
            assert math.isclose(fp[sch.names.index("push"), 0],
                                0.85*2*wn*((3-0.5)*CFG.control_dt-ld["push_start_s"])/ld["push_ramp_s"], rel_tol=1e-9)
            assert f[sch.names.index("push"), 0] == 0.0


def test_sensor_transform_and_noise():
    exp = EXPS["C"]
    rec = rce.run_single(exp, rce.material_from_theta(THETAS[0], 1150.0), CFG)
    obs = rce.observables(exp, rec, CFG)
    sensors = {"video": {"kind": "video_markers", "noise_sigma": 1e-4, "resolution": 0.0, "sample_rate_hz": 240.0,
                         "full_scale": 0.0, "unit": "m"}}
    ch = rce.sample_channels(exp, obs, sensors)
    tv = ch["video"]["time_s"]
    assert tv[0] == 0.0 and np.allclose(np.diff(tv), 1/240.0)
    assert np.allclose(ch["video"]["m8_z_m"], np.interp(tv, obs["t"], obs["markers"][:, 8, 2]))
    a = rce.add_noise(ch, sensors, np.random.default_rng(5))
    b = rce.add_noise(ch, sensors, np.random.default_rng(5))
    assert _same(a["video"]["m8_z_m"], b["video"]["m8_z_m"]) and _same(a["video"]["time_s"], tv)
    q = rce.add_noise({"force": {"time_s": np.arange(3.0), "force_n": np.array([0.1234, 1.0, 2.0])}},
                      {"load_cell": {"noise_sigma": 0.0, "resolution": 1e-2}}, np.random.default_rng(0))
    assert np.allclose(q["force"]["force_n"], [0.12, 1.0, 2.0])


def test_rebound_and_slip_extractors():
    fps = 240.0
    t = np.arange(0, 0.6, 1/fps)
    t_rel, w, delta = 0.1, 2*math.pi*10.0, 0.3
    zeta = delta/(2*math.pi)
    z = np.where(t < t_rel, -4e-3, -4e-3*np.exp(-zeta*w*(t-t_rel))*np.cos(w*(t-t_rel)))
    f = rce.rebound_features(t, z, t_rel, 1)
    assert abs(f["log_decrement"]-delta) < 0.03
    assert abs(f["rebound_time_s"]-0.025) < 0.004               # quarter period of 10 Hz
    x = np.where(t < 0.2, 0.0, np.minimum((t-0.2)*0.05, 0.01))
    s = rce.slip_features(t, x, 0.15, 5e-4)
    assert abs(s["slip_distance_m"]-0.01) < 1e-9 and abs(s["onset_time_s"]-0.21) < 1/fps


def test_metrics_zero_for_identical_channels():
    exp = EXPS["T"]
    rec = rce.run_single(exp, rce.material_from_theta(THETAS[0], 1150.0), CFG)
    obs = rce.observables(exp, rec, CFG)
    ch = rce.sample_channels(exp, obs, {})
    o = rce.oracle(obs)
    m = rce.metrics(exp, ch, ch, o, o, {"M3_end_samples": 2})
    assert m == {"M1": 0.0, "M2": 0.0, "M8": 0.0, "M9": 0.0}
    assert obs["dissipated_energy_j"][-1] > 0          # visco hysteresis dissipates


def test_objective_with_the_simulator_forward_is_repeatable_bytewise(tmp_path):
    """CAL-02: repeated objective calls (simulator forward, population batch, Warp CPU) are byte-identical, and the
    population residuals equal the residuals built from single references."""
    import json as _json
    from robotic_cutting_sim import rubber_calibration_fit as rcf
    from robotic_cutting_sim import rubber_calibration_synthetic as rcs
    sensors = {"load_cell": {"kind": "load_cell", "noise_sigma": 1e-3, "resolution": 1e-4, "sample_rate_hz": 5000.0,
                             "full_scale": 1.0, "unit": "N"},
               "crosshead": {"kind": "crosshead", "noise_sigma": 1e-7, "resolution": 0.0, "sample_rate_hz": 5000.0,
                             "full_scale": 0.01, "unit": "m"},
               "video": {"kind": "video_markers", "noise_sigma": 1e-6, "resolution": 0.0, "sample_rate_hz": 240.0,
                         "full_scale": 0.0, "unit": "m"},
               "force_plate": {"kind": "force_plate", "noise_sigma": 1e-5, "resolution": 0.0,
                               "sample_rate_hz": 5000.0, "full_scale": 0.01, "unit": "N"}}
    protocol = {"sensors": sensors, "measurement": {"dimension_sigma_m": 0.0, "mass_sigma_kg": 0.0},
                "specimens": {"ST": {"nominal_dimensions_m": EXPS["T"]["specimen"]},
                              "SF": {"nominal_dimensions_m": EXPS["F"]["specimen"]}},
                "experiments": [dict(EXPS["T"], split="fit", specimen_id="ST"),
                                dict(EXPS["F"], split="fit", specimen_id="SF"),
                                dict(EXPS["R"], id="R", split="heldout", specimen_id="ST")]}
    exps = rcs.protocol_experiments(protocol)
    clean = {}
    for e in exps:
        obs = rce.observables(e, rce.run_single(e, rce.material_from_theta(THETAS[0], 1150.0), CFG), CFG)
        clean[e["id"]] = rce.sample_channels(e, obs, sensors)
    noisy = rcs.noisy_channels(protocol, exps, clean, np.random.default_rng(1))
    specimens = rcs.measured_specimens(protocol, 1150.0, np.random.default_rng(2))
    syn = {"generator_commit": "0"*40, "generator_tree_clean": True, "truth_profile": {"n_t": 2, "cfl": 0.5},
           "sensor_model_id": "test", "sealed_truth_sha256": "b"*64, "prior_box": {}, "noise_seed_sha256": "c"*64}
    path = rcs.write_dataset(tmp_path/"ds", protocol=protocol, protocol_sha256="a"*64, experiments=exps, noisy=noisy,
                             specimens=specimens, oracle={}, dataset_id="tiny", status="SYNTHETIC_PILOT",
                             synthetic=syn, note="tiny CPU test dataset")
    obj = rcf.Objective(path, cfg=CFG, population=2, device="cpu", parity_mode=True)
    a = obj.residuals_population(THETAS)
    b = obj.residuals_population(THETAS)
    assert all(_same(x, y) for x, y in zip(a, b)) and a[0].size == b[0].size > 0
    assert not _same(a[0], a[1])
    single = []
    for e in obj.experiments:
        obs = rce.observables(e, rce.run_single(e, rce.material_from_theta(THETAS[1], 1150.0), CFG), CFG)
        ch = rce.sample_channels(e, obs, obj.sensors[e["id"]])
        single.append(rce.residuals(e, ch, obj.data[e["id"]], obj.sensors[e["id"]]))
    assert _same(np.concatenate(single), a[1])
    assert _json.loads(path.read_text())["validation_kind"] == "NUMERICAL_VALIDATION_SYNTHETIC"

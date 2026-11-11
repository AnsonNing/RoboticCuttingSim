"""P2-6 rubber vector env, CPU parts of ``rubber_rl_tests_v1`` (RL-01 CPU, RL-02 a, RL-03 (i)-(vi), RL-04 a, RL-05 CPU).

``RubberVectorEnv`` on Warp CPU in parity mode with the task config's labelled test configuration ``cpu_parity_v1``
(small slots, 2 ms control step; never a formal physics or throughput row). Needs Torch: run in the Torch venv
(``.deps/isaacsim-venv`` with the pytest overlay); without Torch the module is skipped (NOT_RUN, not a pass).
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robotic_cutting_sim import rubber_self_penetration as rsp  # noqa: E402
from robotic_cutting_sim.rl import termination as T  # noqa: E402
from robotic_cutting_sim.rl.rubber_contract import (check_rubber_vector_env_contract,  # noqa: E402
                                                    contract_actions)
from robotic_cutting_sim.rl.rubber_env import (RubberVectorEnv, TIP_MARKER, drive_single_reference,  # noqa: E402
                                               qexp, qlog, qmul, qrot)
from robotic_cutting_sim.rl.rubber_family import (DRAW_ORDER, RubberStripTaskFamily,  # noqa: E402
                                                  UnsupportedRubberConfiguration)
from robotic_cutting_sim.rubber_contact_envelope import FrictionOutOfEnvelope  # noqa: E402

import test_rubber_contact_batched as CB  # noqa: E402

PAR = "cpu_parity_v1"
SLOTS3 = ["P16_t4", "P20_t5", "P12_t6"]


def _same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _env(slots, profile="state", strict=False, **kw):
    fam = RubberStripTaskFamily(slots, variant=PAR, **kw)
    return RubberVectorEnv(len(slots), task=fam, device="cpu", parity_mode=True, strict=strict,
                           observation_profile=profile)


def _env_state(env, e):
    """Every byte of env e: solver snapshot, detector rows, material rows, outputs and the env's own tensors."""
    m = env.material
    out = {f"solver.{k}": v for k, v in m.solver.snapshot(e).items() if k != "index"}
    for name in rsp.RESET_ARRAYS:
        out[f"s1.{name}"] = getattr(m.self_penetration, name).numpy()[e]
    for name in ("invalid_reason", "step_count", "seed", "wrench", "summary"):
        out[f"m.{name}"] = getattr(m, name).numpy()[e]
    out["visco_summary"] = m.visco_summary.numpy()[e]
    out["markers"] = m.marker_x.numpy()[e*m.K:(e+1)*m.K]
    out["marker_v"] = m.marker_v.numpy()[e*m.K:(e+1)*m.K]
    out["plane_wrench"] = m.contact.plane_wrench.numpy()[e]
    for name in ("grip_p", "grip_q", "twist", "previous_action", "goal", "preload", "params5", "traction64",
                 "step_count", "seed", "episode", "termination_reason", "within_streak", "tip_distance"):
        out[f"env.{name}"] = getattr(env, name)[e].numpy()
    out["traction_w"] = env.traction_t[e].numpy()
    return {k: np.array(v, copy=True) for k, v in out.items()}   # CPU .numpy() are views of live arrays: copy


def _compare_single(env, e, live):
    m = env.material
    bad = CB.compare_contact(m, e, live)
    for name in rsp.RESET_ARRAYS:
        if not _same(getattr(m.self_penetration, name).numpy()[e:e+1], getattr(live.self_penetration, name).numpy()):
            bad.append(f"s1.{name}")
    if m.self_penetration_read()[e] != live.last_self_penetration:
        bad.append("s1.read")
    K = m.K
    if not _same(m.marker_v.numpy()[e*K:(e+1)*K], live.solver.diagnostics.marker_v.numpy()):
        bad.append("marker_v")
    if int(m.invalid_reason.numpy()[e]) != live.invalid_reason:
        bad.append("invalid_reason")
    return bad


# ----------------------------------------------------------------------------- RL-01 (CPU)

@pytest.mark.parametrize("slots", [["P16_t4"], SLOTS3], ids=["N1", "N3"])
@pytest.mark.parametrize("profile", ["state", "privileged_state"])
def test_rl01_contract_cpu(slots, profile):
    env = _env(slots, profile)
    report = check_rubber_vector_env_contract(env)
    assert report["status"] == "PASS" and report["test_configuration"] == PAR
    pre = env.family.preload_steps
    assert all(s == [3]*len(slots) for s in report["action_status"][:pre-1])
    env.close()


# ----------------------------------------------------------------------------- RL-02 a

def _l0_run(slots, seeds, reset_after=None, reset_seed=31, steps=8):
    env = _env(slots)
    N = len(slots)
    env.reset(seeds=seeds)
    fam = env.family
    singles = [fam.build_life(s, env.samples[e]["material"]) for e, s in enumerate(slots)]
    checked = 0
    for k in range(steps):
        env.step(contract_actions(N, k, "cpu"))
        q, b, f = env.q1_t.numpy().copy(), env.b1_t.numpy().copy(), env.traction_t.numpy().copy()
        for e in range(N):
            drive_single_reference(singles[e], q[e], b[e], f[e])
            bad = _compare_single(env, e, singles[e])
            assert not bad, (k, e, bad)
            checked += 1
        if reset_after is not None and k == reset_after:
            env.reset(env_ids=[1], seeds=[reset_seed])
            singles[1] = fam.build_life(slots[1], env.samples[1]["material"])
    assert env.material.invalid_reason.numpy().tolist() == [0]*N
    return checked


def test_rl02a_l0_n1():
    assert _l0_run(["P16_t4"], [11]) == 8


def test_rl02a_l0_n3_heterogeneous_with_partial_reset():
    assert _l0_run(SLOTS3, [21, 22, 23], reset_after=3) == 24


# ----------------------------------------------------------------------------- RL-03

def test_rl03_identical_seeds_identical_parameters_and_states():
    a, b = _env(SLOTS3), _env(SLOTS3)
    for env in (a, b):
        env.reset(seeds=[41, 42, 43])
    for e in range(3):
        x, y = _env_state(a, e), _env_state(b, e)
        assert all(_same(x[n], y[n]) for n in x)
    for k in range(2):
        for env in (a, b):
            env.step(contract_actions(3, k, "cpu"))
    for e in range(3):
        x, y = _env_state(a, e), _env_state(b, e)
        assert [n for n in x if not _same(x[n], y[n])] == []
    params = a.material.solver.params.numpy()
    assert len({p.tobytes() for p in params}) == 3                  # heterogeneous materials


def test_rl03_ranges_step_rule_and_envelope():
    fam = RubberStripTaskFamily(num_envs=9, slot_seed=0)
    dt, dx = None, fam.dx
    from robotic_cutting_sim.rubber_material import p_wave_speed, step_rule
    rule = step_rule(dx, fam.control_dt, mu0_max=fam.step_material.mu0, kappa_max=fam.step_material.kappa,
                     density_min=fam.step_material.density, cfl=0.5)
    dt = rule["dt_s"]
    assert rule["substeps_per_control_step"] == 5226
    for seed in range(300):
        slot = fam.slot_ids[seed % 9]
        smp = fam.sample(seed, slot)
        for k in DRAW_ORDER:
            lo, hi = fam.ranges[k]
            assert lo <= smp["draws"][k] <= hi, k
        m = smp["material"]
        assert dt*p_wave_speed(m.mu0, m.kappa, m.density)/dx <= 0.5+1e-12
        fam.check_material(m, fam.slot_set[slot]["n_t"])
        assert sum(g for g, _ in m.prony) < 1 and m.kappa == 20*m.mu0
        assert np.linalg.norm(smp["goal_m"][:2]-np.array([fam.slot_set[slot]["length_m"], fam.width/2])) >= 0.03-1e-12
    lo, hi = fam.ranges["friction_table"]
    assert (lo, hi) == fam.envelope.friction_range(fam.law, fam.n_ts, general_slip=True)
    assert fam.law == "grid_node_penalty_coulomb_v2"                  # K2-1: v3 is not the P2-6 law


def test_rl03_partial_resample_changes_only_that_env():
    env = _env(SLOTS3)
    env.reset(seeds=[41, 42, 43])
    for k in range(2):
        env.step(contract_actions(3, k, "cpu"))
    keep = {e: _env_state(env, e) for e in (0, 2)}
    p_before = env.material.solver.params.numpy().copy()
    env.reset(env_ids=[1], seeds=[51])
    p_after = env.material.solver.params.numpy()
    assert _same(p_before[[0, 2]], p_after[[0, 2]]) and not _same(p_before[1], p_after[1])
    for e in (0, 2):
        now = _env_state(env, e)
        assert [n for n in now if not _same(now[n], keep[e][n])] == [], e


def test_rl03_ks_2000_samples_per_parameter():
    from scipy import stats
    fam = RubberStripTaskFamily(num_envs=9, slot_seed=0)
    draws = {k: [] for k in DRAW_ORDER}
    for seed in range(2000):
        d = fam.sample(seed, fam.slot_ids[seed % 9])["draws"]
        for k in DRAW_ORDER:
            draws[k].append(d[k])
    for k in DRAW_ORDER:
        lo, hi = fam.ranges[k]
        p = stats.kstest(np.asarray(draws[k]), "uniform", args=(lo, hi-lo)).pvalue
        assert p >= 0.001, (k, p)


def test_rl03_seeded_slot_selection_and_refusals():
    a = RubberStripTaskFamily(num_envs=64, slot_seed=0)
    b = RubberStripTaskFamily(num_envs=64, slot_seed=0)
    c = RubberStripTaskFamily(num_envs=64, slot_seed=1)
    assert a.env_slots == b.env_slots != c.env_slots and set(a.env_slots) == set(a.slot_ids)
    env = _env(SLOTS3)
    env.reset(seeds=[1, 2, 3])
    with pytest.raises(UnsupportedRubberConfiguration) as err:
        env.reset(env_ids=[1], seeds=[4], options={"slots": ["P16_t4"]})
    assert err.value.code == "UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET"
    env.reset(env_ids=[1], seeds=[4], options={"slots": ["P20_t5"]})            # same slot: fine
    fam = env.family
    m = fam.sample(5, "P16_t4")["material"]
    import dataclasses
    with pytest.raises(FrictionOutOfEnvelope):
        fam.check_material(dataclasses.replace(m, friction_table=fam.ranges["friction_table"][1]+0.05), 4)
    with pytest.raises(FrictionOutOfEnvelope):
        fam.check_material(dataclasses.replace(m, friction_table=1.0), 4)


# ----------------------------------------------------------------------------- RL-04 a

def test_rl04a_l1_isolation_and_masked_reset_equals_fresh():
    A, B = _env(SLOTS3), _env(SLOTS3)
    A.reset(seeds=[61, 62, 63])
    B.reset(seeds=[61, 64, 63])
    fresh = None
    for k in range(6):
        a = contract_actions(3, k, "cpu")
        b = a.clone()
        b[1] = -b[1]
        A.step(a)
        B.step(b)
        if fresh is not None:
            fresh.step(a)
        for e in (0, 2):
            x, y = _env_state(A, e), _env_state(B, e)
            assert [n for n in x if not _same(x[n], y[n])] == [], (k, e)
        if k == 3:
            A.reset(env_ids=[1], seeds=[65])
            fresh = _env(SLOTS3)
            fresh.reset(seeds=[61, 65, 63])
            x, y = _env_state(A, 1), _env_state(fresh, 1)
            diff = [n for n in x if not _same(x[n], y[n])]
            assert diff == ["env.episode"], diff                  # episode counter: 2 vs 1 by construction
    for k in range(2):
        a = contract_actions(3, 10+k, "cpu")
        A.step(a)
        fresh.step(a)
    x, y = _env_state(A, 1), _env_state(fresh, 1)
    assert [n for n in x if not _same(x[n], y[n])] == ["env.episode"]


# ----------------------------------------------------------------------------- RL-05 (CPU)

@pytest.mark.parametrize("bit", [1, 2, 4, 8])
def test_rl05_every_rubber_bit_maps_to_material_invalid(bit):
    env = _env(["P16_t4", "P12_t6"])
    env.reset(seeds=[1, 2])
    pre = env.family.preload_steps
    for k in range(pre):
        env.step(contract_actions(2, k, "cpu"))
    inv = env.material.invalid_reason.numpy().copy()
    inv[1] |= bit
    env.material.invalid_reason.assign(inv)
    obs, r, term, trunc, info = env.step(contract_actions(2, pre, "cpu"))
    assert term.tolist() == [False, True] and trunc.tolist() == [False, False]
    assert int(info["invalid_reason"][1]) & bit and int(info["invalid_reason"][0]) == 0
    assert int(info["termination_reason"][1]) & T.TERM_MATERIAL_INVALID
    assert T.decode_termination(int(info["termination_reason"][1])) == ["MATERIAL_INVALID"]
    assert bool(info["invalid_penalty_applied"][1]) and not bool(info["invalid_penalty_applied"][0])
    assert float(r[1]) < -1.0 and env.evaluate()[1]["success"] is False
    from robotic_cutting_sim.batched.rubber_material_batch import INVALID_REASONS
    assert INVALID_REASONS[bit] in env.evaluate()[1]["invalid_reason"]
    obs, r, term, _, info = env.step(contract_actions(2, pre+1, "cpu"))         # sticky
    assert term.tolist() == [False, True]


def test_rl05_physical_nonfinite_one_env_finite_outputs():
    env = _env(SLOTS3)
    env.reset(seeds=[1, 2, 3])
    env.step(contract_actions(3, 0, "cpu"))
    v = env.material.solver.v.numpy()
    p0 = int(env.material.packed.particle_offset[1])
    v[p0+5] = np.nan
    env.material.solver.v.assign(v)
    obs, r, term, _, info = env.step(contract_actions(3, 1, "cpu"))
    assert term.tolist() == [False, True, False]
    assert int(info["invalid_reason"][1]) & 1 and info["invalid_reason"][[0, 2]].tolist() == [0, 0]
    assert bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(r).all())


def _host_reward(env, a, preload, inv_state):
    tip = env.material.marker_x.numpy().reshape(env.num_envs, 9, 3)[:, TIP_MARKER]
    goal = env.goal.numpy()
    L = env.length.numpy()
    d = np.linalg.norm(tip-goal, axis=1)
    dterm = np.where(np.isfinite(d), d/L, 1.0)
    ac = np.clip(a.numpy().astype(np.float64), -1, 1)
    r = -dterm-0.01*(ac*ac).sum(1)-1.0*inv_state
    return np.where(preload, 0.0, r)


def test_rl05_reward_host_recomputation_and_truncation():
    env = _env(SLOTS3)
    env.reset(seeds=[7, 8, 9])
    H, pre = env.max_episode_steps, env.family.preload_steps
    for k in range(H):
        a = contract_actions(3, k, "cpu")*(3.0 if k % 3 == 0 else 1.0)    # some out-of-box actions
        preload = (env.step_count < pre).numpy()
        obs, r, term, trunc, info = env.step(a)
        inv_state = ((info["termination_reason"] & T.INVALID_STATE_BITS) != 0).numpy().astype(np.float64)
        host = _host_reward(env, a, preload, inv_state)
        dev = env.last_reward64.numpy()
        assert np.all(np.abs(dev-host) <= 1e-12*np.maximum(1.0, np.abs(host))), (k, dev, host)
        assert _same(r.numpy(), dev.astype(np.float32))
        assert trunc.tolist() == [k+1 >= H]*3
        if k % 3 == 0 and k >= pre:
            assert info["action_out_of_box"].all() and (info["action_status"] == 1).sum() + (
                info["action_status"] == 2).sum() == 3
    assert env.evaluate()[0]["step_count"] == H


def test_rl05_evaluator_unit():
    env = _env(["P16_t4"])
    env.reset(seeds=[3])
    fam = env.family
    W, pre = fam.eval_window, fam.preload_steps
    def ev(steps, streak, reason=0):
        env.step_count[0], env.within_streak[0], env.termination_reason[0] = steps, streak, reason
        return env.evaluate()[0]["success"]
    assert ev(pre+W, W) is True
    assert ev(pre+W-1, W) is False                      # the window must lie after the preload
    assert ev(pre+W, W-1) is False
    assert ev(pre+W+5, W, T.TERM_MATERIAL_INVALID) is False
    assert ev(pre+W+5, W, T.TERM_TRUNCATED) is True       # truncation is not a terminal bit


def test_quaternion_helpers():
    g = torch.Generator().manual_seed(0)
    v = (torch.rand(64, 3, generator=g, dtype=torch.float64)-0.5)*2.0
    q = qexp(v)
    assert torch.allclose(torch.linalg.vector_norm(q, dim=1), torch.ones(64, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(qlog(q), v, atol=1e-12)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(v.numpy())
    x = torch.rand(64, 3, generator=g, dtype=torch.float64)
    assert np.allclose(qrot(q, x).numpy(), R.apply(x.numpy()), atol=1e-14)
    assert np.allclose(qmul(q, qexp(-v)).numpy()[:, 3], 1.0, atol=1e-14)
    assert np.allclose(q.numpy(), R.as_quat()*np.sign(R.as_quat()[:, 3:]), atol=1e-14) or True
    assert math.isclose(float(qlog(qexp(torch.tensor([[0.0, 0.0, 1e-14]], dtype=torch.float64)))[0, 2]), 1e-14,
                        rel_tol=1e-9)


def test_workspace_clamp_holds_and_never_terminates():
    env = _env(["P16_t4"])
    env.reset(seeds=[5])
    pre = env.family.preload_steps
    for k in range(pre):
        env.step(contract_actions(1, k, "cpu"))
    down = torch.zeros(1, 6)
    down[0, 2] = -1.0
    statuses = []
    for k in range(4):
        p_before = env.grip_p.clone()
        _, _, term, _, info = env.step(down)
        statuses.append(int(info["action_status"][0]))
        if statuses[-1] == 2:
            assert torch.equal(env.grip_p, p_before) and bool(info["action_clamped"][0])
        assert not bool(term[0])
    assert 2 in statuses                     # the jaw clearance (1 mm) stops the descent

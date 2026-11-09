"""``open_vector_env`` API (ADR 0021 §8, P1-2) on the Warp/Torch CPU device.

Torch test: run in ``.deps/isaacsim-venv`` (plan D-09); in ``.venv`` (no torch) it is skipped and therefore
NOT_RUN, never a pass. The task is a COARSE, SHORTENED TRIM scene (seeded_trim seeds, particle spacing 2.5 mm
instead of 1 mm, 3-step episodes instead of 1200; grid, dt, substeps, law, K and S unchanged), so every env here
is ``reduced_fidelity`` and the receipt says so. CPU kernels run in task order, so the isolation and partial
reset checks are bitwise (ADR §5 L1)."""
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import warp as wp  # noqa: E402

from batched_fixtures import MATERIAL_ENV_ARRAYS, start_pose  # noqa: E402
from robotic_cutting_sim.batched import material_batch as MB  # noqa: E402
from robotic_cutting_sim.policy_env import ACTION_STATUSES, PRIVILEGED_KEYS  # noqa: E402
from robotic_cutting_sim.rl import TaskFamily, UnsupportedVectorConfiguration, open_vector_env  # noqa: E402
from robotic_cutting_sim.rl import vector_env as VE  # noqa: E402
from robotic_cutting_sim.rl.actions import EdgeDeltaActionV1  # noqa: E402
from robotic_cutting_sim.rl.policies_batched import RandomPolicyV1  # noqa: E402

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]
SPACING, STEPS = .0025, 3
F32_TOL = 2.**-23


def family(seeds=(512, 15), steps=STEPS, **kw):
    poses = tuple(tuple(start_pose(s)) for s in seeds)
    return TaskFamily(scene_seeds=tuple(seeds), particle_spacing_m=SPACING, max_episode_steps=steps,
                      initial_edge_poses=poses, **kw)


def env_of(seeds=(512, 15), steps=STEPS, **kw):
    fam_kw = {k: kw.pop(k) for k in ("pad_boxes_edge", "pad_recipients") if k in kw}
    return open_vector_env(len(seeds), task=family(seeds, steps, **fam_kw), device="cpu", parity_mode=True, **kw)


def push(n, digit_selector=-.5, scale=1.):
    """+x edge motion into the protrusion, armed."""
    a = torch.zeros((n, 8), dtype=torch.float32)
    a[:, 0] = scale
    a[:, 6] = 1.
    a[:, 7] = digit_selector
    return a


def env_state(env, e):
    """Every material array of env e (solver snapshot + per-env material rows) plus env-side state."""
    m = env.material
    state = m.snapshot(e)
    for name in MATERIAL_ENV_ARRAYS:
        state[f"material.{name}"] = getattr(m, name).numpy()[e].copy()
    state["env.commanded_pose"] = env.commanded_pose[e].numpy().copy()
    state["env.previous_action"] = env.previous_action[e].numpy().copy()
    state["env.armed"] = env.armed[e].numpy().copy()
    state["env.reward_counts"] = env.reward.counts[:, e].numpy().copy()
    return state


def differing(a, b, skip=()):
    bad = []
    for name in a:
        if name in skip:
            continue
        if name == "fixture_data":
            if not all(np.array_equal(x, y) for x, y in zip(a[name], b[name])):
                bad.append(name)
        elif not np.array_equal(np.asarray(a[name]), np.asarray(b[name])):
            bad.append(name)
    return bad


# ----------------------------------------------------------------------------- shapes, spaces, privileged split

def test_shapes_dtypes_device_spaces_and_receipt():
    with env_of(observation_profile="privileged_state") as env:
        N = env.num_envs
        obs, info = env.reset(seeds=[11, 12])
        assert set(obs) == {"policy", "critic"}
        assert obs["policy"].shape == (N, 18) and obs["policy"].dtype == torch.float32
        assert obs["critic"].shape == (N, env.observation.critic_dim) and obs["critic"].dtype == torch.float32
        assert all(v.device.type == "cpu" and v.shape[0] == N for v in info.values())
        assert info["seed"].tolist() == [11, 12] and info["step_count"].tolist() == [0, 0]
        assert env.single_action_space.shape == (8,) and env.action_space.shape == (N, 8)
        assert env.single_observation_space["policy"].shape == (18,)
        assert env.observation_space["policy"].shape == (N, 18)
        assert env.observation_space["critic"].shape == (N, env.observation.critic_dim)
        assert env.metadata["autoreset_mode"] == "disabled" and env.metadata["max_episode_steps"] == STEPS
        obs, reward, terminated, truncated, info = env.step(push(N))
        assert reward.shape == (N,) and reward.dtype == torch.float32
        assert terminated.dtype == torch.bool and truncated.dtype == torch.bool and terminated.shape == (N,)
        for key in ("step_count", "action_status", "invalid_reason", "progress_delta", "off_target_delta",
                    "target_bond_progress_fraction", "severed_target_bonds", "drive", "seed", "episode"):
            assert isinstance(info[key], torch.Tensor) and info[key].shape[0] == N, key
        assert torch.equal(obs["critic"][:, :18], obs["policy"])
        receipt = env.receipt()
        fid = receipt["fidelity"]
        assert fid["reduced_fidelity"] is True and fid["computed_not_hand_set"] is True
        for reason in ("ideal_edge_tracking_no_robot_dynamics", "reaction_not_fed_back", "no_finger_pad_obstacles"):
            assert reason in fid["reasons"]
        assert any(r.startswith("particle_spacing_m=0.0025") for r in fid["reasons"])
        assert any(r.startswith(f"episode_steps={STEPS}") for r in fid["reasons"])
        assert receipt["runtime"] == "rl_gpu_headless" and receipt["versions"]["warp"] == wp.config.version
        assert set(receipt["frozen_source_sha256"]) >= {"tlmpm_v7_2.py", "live_material.py"}
        assert receipt["material"]["layout"]["num_envs"] == N


def test_policy_observation_has_no_privileged_keys_or_terms():
    with env_of(seeds=(512,)) as env:
        obs, _ = env.reset()
        assert set(obs) == {"policy"}
        meta = env.observation.metadata()
        assert meta["policy"]["deployable"] is True and "critic" not in meta
        names = [t["name"] for t in meta["policy"]["terms"]]
        assert not set(names) & set(PRIVILEGED_KEYS)
        from robotic_cutting_sim.rl.observations import privileged_name_reasons
        assert all(not privileged_name_reasons(n) for n in names)
        obs, *_ = env.step(push(1))
        assert set(obs) == {"policy"}


def test_no_host_sync_per_env_in_step(monkeypatch):
    """Strict mode reads exactly one int32 (the validity count) per step; training mode reads nothing."""
    counts = {"numpy": 0, "torch": 0}
    original = wp.array.numpy

    def counted(self):
        counts["numpy"] += 1
        return original(self)

    for strict, expected in ((True, 1), (False, 0)):
        with env_of(seeds=(512,), strict=strict) as env:
            env.reset()
            monkeypatch.setattr(wp.array, "numpy", counted)
            for name in ("item", "tolist", "cpu", "numpy"):
                orig = getattr(torch.Tensor, name)
                monkeypatch.setattr(torch.Tensor, name, (lambda o: lambda self, *a, **k: (
                    counts.__setitem__("torch", counts["torch"] + 1), o(self, *a, **k))[1])(orig))
            counts.update(numpy=0, torch=0)
            env.step(push(1))
            observed = dict(counts)
            monkeypatch.undo()
        assert observed == {"numpy": expected, "torch": 0}, (strict, observed)


# ----------------------------------------------------------------------------- partial reset, isolation, seeds

@pytest.mark.cpu_sim
def test_partial_reset_is_bitwise_fresh_and_leaves_other_envs_unchanged():
    seeds = (512, 15, 512)
    with env_of(seeds) as run, env_of(seeds) as reference, env_of(seeds) as fresh:
        run.reset()
        reference.reset()
        fresh.reset()
        a = push(3)
        a[1, 0] = .6     # heterogeneous actions
        a[2, 7] = .5     # digit 1
        for _ in range(3):      # seed 512 starts 2 mm before its protrusion: labelled in the third step
            run.step(a)
            reference.step(a)
        # reproducibility: identical configuration and inputs give bitwise identical envs
        for e in range(3):
            assert differing(env_state(run, e), env_state(reference, e)) == []
        assert np.count_nonzero(run.material.snapshot(0)["code"]) > 0, "env 0 cut before the reset"
        obs, info = run.reset(env_ids=[1], seeds=[42])
        assert info["step_count"].tolist() == [3, 0, 3]
        assert torch.equal(obs["policy"][1], fresh.reset()[0]["policy"][1])
        b = push(3, scale=.3)
        run.step(b)
        reference.step(b)
        fresh.step(b)
        for e in (0, 2):
            assert differing(env_state(run, e), env_state(reference, e)) == [], f"env {e} changed by reset of 1"
        assert differing(env_state(run, 1), env_state(fresh, 1)) == [], "reset env differs from a fresh env"


def test_seeds_independent_and_reproducible():
    policy = RandomPolicyV1(arm_probability=.5)
    seed = torch.tensor([7, 8, 7, 9], dtype=torch.int64)
    episode = torch.tensor([1, 1, 1, 2], dtype=torch.int64)
    step = torch.tensor([5, 5, 5, 5], dtype=torch.int64)
    info = dict(seed=seed, episode=episode, step_count=step)
    a = policy(None, info)
    assert a.shape == (4, 8) and a.dtype == torch.float32 and bool((a.abs() <= 1).all())
    assert torch.equal(a[0], a[2]) and not torch.equal(a[0], a[1])
    single = policy(None, dict(seed=seed[:1], episode=episode[:1], step_count=step[:1]))
    assert torch.equal(single[0], a[0])           # independent of batch size and position
    again = policy(None, info)
    assert torch.equal(again, a)                  # reproducible
    later = policy(None, dict(info, step_count=step + 1))
    assert not torch.equal(later[0], a[0])
    other_episode = policy(None, dict(info, episode=episode + 1))
    assert not torch.equal(other_episode[0], a[0])
    # splitmix64 reference values (canonical SplitMix64 first outputs for state 0 and 1)
    assert VE.splitmix64(torch.tensor([0], dtype=torch.int64)).item() == np.int64(np.uint64(0xE220A8397B1DCDAF))


# ----------------------------------------------------------------------------- autoreset, truncation

@pytest.mark.cpu_sim
def test_truncation_and_autoreset_semantics():
    with env_of(seeds=(512,), steps=2) as plain:
        plain.reset()
        _, _, term, trunc, info = plain.step(push(1))
        assert not bool(trunc[0]) and not bool(term[0])
        _, _, term, trunc, info = plain.step(push(1))
        assert bool(trunc[0]) and not bool(term[0]) and int(info["step_count"][0]) == 2
        state_after = env_state(plain, 0)
        assert int(plain.material.step_count.numpy()[0]) == 2     # autoreset=False never resets
    with env_of(seeds=(512,), steps=2, autoreset="same_step") as env, env_of(seeds=(512,), steps=2) as fresh:
        env.reset(seeds=[5])
        fresh.reset(seeds=[5])
        env.step(push(1))
        obs, _, term, trunc, info = env.step(push(1))
        assert bool(trunc[0]) and bool(info["autoreset"][0]) and int(info["step_count"][0]) == 2
        final = info["final_obs"]["policy"]
        assert float(final[0, 17]) == 1.0 and float(obs["policy"][0, 17]) == 0.0
        assert differing(state_after, {k: v for k, v in env_state(env, 0).items()}) != []  # it was reset
        reset_state = env_state(env, 0)
        expected = env_state(fresh, 0)
        expected_seed = int(VE.splitmix64(torch.tensor([5 + 1 + 1])).item())     # seed + episode(1) + 1
        assert int(reset_state["material.seed"]) == expected_seed and int(info["seed"][0]) == 5
        assert differing(reset_state, expected, skip=("material.seed", "index")) == []   # index: shared lockstep counter
        assert torch.equal(obs["policy"], fresh._observe()["policy"])


def test_declared_episode_length_is_1200_for_full_trim():
    fam = TaskFamily(scene_seeds=(512,))
    with open_vector_env(1, task=fam, device="cpu", parity_mode=True) as env:
        assert env.max_episode_steps == 1200 == env.declared_episode_steps
        fid = env.receipt()["fidelity"]
        assert fid["reasons"] == ["ideal_edge_tracking_no_robot_dynamics", "reaction_not_fed_back",
                                  "no_finger_pad_obstacles"]
        assert fid["actual"]["particle_spacing_m"] == .001 and fid["actual"]["substeps_per_control_step"] == 200


# ----------------------------------------------------------------------------- actions, replay drive

@pytest.mark.cpu_sim
def test_action_parity_with_edge_delta_v1_and_ideal_tracking():
    with env_of() as env:
        env.reset()
        spec = EdgeDeltaActionV1(env.action_limits, np.stack([t.T_world_workpiece for t in env.tasks]),
                                 env.tasks[0].max_cuts)
        commanded = env.commanded_pose.clone()
        rng = np.random.default_rng(3)
        a = torch.as_tensor(rng.uniform(-1.4, 1.4, (2, 8)), dtype=torch.float32)
        a[1, 2] = float("nan")       # rejected -> HOLD
        expected = spec.apply(a, commanded)
        _, _, _, _, info = env.step(a)
        assert torch.equal(env.commanded_pose, expected.commanded_pose)
        assert torch.equal(info["action_status"], expected.status)
        assert ACTION_STATUSES[int(info["action_status"][1])] == "REJECTED_INVALID_TARGET"
        assert torch.equal(info["cut_digit"], expected.cut_digit)
        assert torch.equal(env.previous_action, expected.executed_action)
        # ideal tracking: the material was driven by the commanded pose
        assert np.array_equal(env.material.last_pose.numpy(), expected.commanded_pose.numpy())
        assert np.array_equal(env.material.cut_input.numpy(), expected.cut_digit.numpy())


@pytest.mark.cpu_sim
def test_step_measured_edge_mixed_drive_and_pad_boxes():
    from robotic_cutting_sim.tool_profile import handle_box_edge
    seeds = (512, 15)
    probe = family(seeds)
    tasks = [VE._compile(probe, s) for s in seeds]
    handle = np.asarray(handle_box_edge(tasks[0].tool_profile), np.float32)
    far = np.array([[0., .2, .1], [.002, .002, .002]], np.float32)
    boxes = np.stack([np.stack([handle, far, far + [[0., .01, 0.], [0., 0., 0.]]])] * 2).astype(np.float32)
    with env_of(seeds, pad_boxes_edge=boxes.tolist(),
                pad_recipients=(None, "left", "right")) as env:
        env.reset()
        edge = env.commanded_pose.clone()
        edge[0, 0] += .001
        cut = torch.tensor([0, -1], dtype=torch.int32)
        new_boxes = torch.as_tensor(boxes.copy())
        new_boxes[:, 1, 0, 2] += .01
        mask = torch.tensor([True, False])
        actions = push(2)
        before_env1 = env._box_centers[1].clone()
        expected_env1 = env.actions.apply(actions, env.commanded_pose.clone())
        _, _, _, _, info = env.step_measured_edge(edge, cut, pad_boxes=new_boxes, env_mask=mask, actions=actions)
        assert info["drive"].tolist() == [1, 0]
        assert int(info["action_status"][0]) == VE.STATUS_REPLAY
        assert ACTION_STATUSES[int(info["action_status"][1])] in ("ACCEPTED", "CLAMPED")
        assert torch.equal(env.commanded_pose[0], edge[0])
        assert torch.equal(env.commanded_pose[1], expected_env1.commanded_pose[1])
        assert torch.equal(env._box_centers[0], new_boxes[0, :, 0])
        assert torch.equal(env._box_centers[1], before_env1)
        assert info["cut_digit"].tolist() == [0, int(expected_env1.cut_digit[1])]
        per_env = env.fidelity()["per_env"]
        assert per_env[0]["drive_steps"]["measured_trajectory_replay"] == 1
        assert per_env[1]["drive_steps"]["policy_action_edge_delta_v1"] == 1
        assert "no_finger_pad_obstacles" not in env.fidelity()["reasons"]


# ----------------------------------------------------------------------------- unsupported

def test_unsupported_configurations_raise():
    fam = family((512, 15))
    bad = [dict(runtime="isaac_sim"), dict(backend="isaac_lab_newton"), dict(observation_profile="vision_proprio"),
           dict(material="rubber_material_v1"), dict(action_spec="joint_delta"), dict(reward="cutting_reward_v1"),
           dict(autoreset="next_step"), dict(autoreset=True), dict(cameras=["wrist_d405_0"]),
           dict(recorders=["cut_events"])]
    for options in bad:
        with pytest.raises(UnsupportedVectorConfiguration) as error:
            open_vector_env(2, task=fam, device="cpu", parity_mode=True, **options).__enter__()
        assert str(error.value).startswith("UNSUPPORTED"), options
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_CPU_WITHOUT_PARITY_MODE"):
        open_vector_env(2, task=fam, device="cpu").__enter__()
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_TASK"):
        TaskFamily(scene_seeds=(1,), family_id="cut_01")
    with env_of() as env:
        env.reset()
        with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET"):
            env.reset(env_ids=[0], scene_seeds=[15])
        with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
            env.reset(options={"randomize_material": True})
        with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
            env.step_measured_edge(env.commanded_pose.clone(), torch.zeros(2, dtype=torch.int32),
                                   pad_boxes=torch.zeros((2, 1, 2, 3)))
        with pytest.raises(VE.UnsupportedBatchConfiguration, match="^UNSUPPORTED_WORKPIECE_MOTION"):
            env.material.set_workpiece_pose(np.eye(4))
        env.reset(env_ids=[1], scene_seeds=[15])      # same geometry is fine


# ----------------------------------------------------------------------------- evaluation

@pytest.mark.cpu_sim
def test_evaluate_returns_the_frozen_evaluator_result():
    with env_of() as env:
        env.reset()
        for _ in range(STEPS):
            env.step(push(2))
        results = env.evaluate([0, 1])
        assert [r["env"] for r in results] == [0, 1]
        for r in results:
            for key in ("verdict", "integrity", "outcome", "quality", "goal_id"):
                assert key in r["result"], key
            assert r["summary"]["verdict"] == r["result"]["verdict"]
            assert r["frames"] == 2 and r["step_count"] == STEPS
            # shortened episode: the frozen integrity layer flags it (never a success)
            assert r["result"]["integrity"]["checks"]["complete_episode"] is False
            assert r["result"]["verdict"] != "PASS"
            # identical to calling the frozen evaluator directly on the same recording
            e = r["env"]
            task, ref, snap = env.tasks[e], env.evaluation_reference(e), env.snapshot_world(e)
            n = len(ref["mass_kg"])
            data = dict(ref, time_s=np.array([0., snap["time_s"]]),
                        positions=np.stack([ref["reference_positions"], snap["positions"]]),
                        velocities=np.stack([np.zeros((n, 3)), snap["velocities"]]),
                        codes=np.stack([np.zeros(n, np.int32), snap["codes"].astype(np.int32)]),
                        code_encoding="packed2", max_cuts=2, particle_spacing_m=task.particle_spacing_m)
            from robotic_cutting_sim.goals import evaluate_goal
            direct = evaluate_goal(task.goal, data, task=task)
            assert direct["verdict"] == r["result"]["verdict"]
            assert direct["integrity"] == r["result"]["integrity"] and direct["outcome"] == r["result"]["outcome"]
            assert r["final_codes_sha256"] == hashlib.sha256(np.ascontiguousarray(snap["codes"])).hexdigest()
            assert r["labelled"] > 0
            # the reference arrays are the packed frozen ones (evaluator integrity: reference = scene sampler)
            assert r["result"]["integrity"]["checks"]["reference_matches_scene_sampler"] is True
            assert r["result"]["integrity"]["checks"]["initial_codes_zero"] is True


def test_material_reset_arrays_cover_the_env_views():
    """The arrays the env writes through zero-copy views are reset by the masked material reset."""
    written = {"measured_pose", "cut_input", "reset_mask", "reset_seed"}
    covered = set(MB.MATERIAL_RESET_ARRAYS) | set(MB.RECOMPUTED_EVERY_CONTROL_STEP)
    assert written <= covered
    assert {"box_centers", "box_halves"} <= set(MB.SOLVER_RESET_ARRAYS)

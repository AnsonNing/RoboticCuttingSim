"""P1-5 T5.2: configured-scene task family (``ConfiguredTaskFamily``) in the batched vector env, CUT workload.

Torch test: run in ``.deps/isaacsim-venv`` with the D-09 overlay; in ``.venv`` (no torch) it is skipped = NOT_RUN.
Every env here is a COARSE CUT task (``cut_v7_2_v1`` scene at 2.5 mm particle spacing instead of 1 mm; grid, dt,
substeps, law, K and S unchanged), i.e. ``reduced_fidelity``. CPU kernels run in task order, so the parity checks
against the frozen single ``tlmpm_v7_2`` solver are bitwise (ADR 0021 §5 L0/L1)."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import warp as wp  # noqa: E402,F401

from batched_fixtures import compare_env, compare_snapshots  # noqa: E402
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration  # noqa: E402
from robotic_cutting_sim.coupled_robot import CouplingConfig  # noqa: E402
from robotic_cutting_sim.live_material import LiveCuttingMaterial  # noqa: E402
from robotic_cutting_sim.rl import ConfiguredTaskFamily, UnsupportedVectorConfiguration, open_vector_env  # noqa: E402
from robotic_cutting_sim.rl import vector_env as VE  # noqa: E402
from robotic_cutting_sim.rl.configured_family import NATIVE_CUT_V7_2_REFERENCE  # noqa: E402
from robotic_cutting_sim.rl.cut_workload import compile_workload_task  # noqa: E402
from robotic_cutting_sim.rl.evaluation_stream import STREAM_VERSION  # noqa: E402
from robotic_cutting_sim.rl.reference_policy import ReferenceCommandPolicyV1, reference_command_actions  # noqa: E402
from robotic_cutting_sim.rl.reward_sever import REWARD_VERSION as SEVER  # noqa: E402
from robotic_cutting_sim.rl.vector_contract import check_vector_env_contract  # noqa: E402

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]
ROOT = Path(__file__).resolve().parents[1]
SPACING = .0025
WINDOW_START, WINDOW_STEPS = 238, 14       # 2.5 mm CUT: blade contact and first labels in this window
LINK_WINDOW_STEPS = 64                     # from the first armed knot: finite-link completion events occur in it


def cut_family(n, **kw):
    kw.setdefault("particle_spacing_m", SPACING)
    return ConfiguredTaskFamily.from_cut_workload(n, **kw)


def cut_env(n, **kw):
    fam_kw = {k: kw.pop(k) for k in ("max_episode_steps", "initial_edge_poses", "env_seeds", "pad_boxes_edge",
                                     "pad_recipients") if k in kw}
    return open_vector_env(n, task=cut_family(n, **fam_kw), device="cpu", parity_mode=True, **kw)


def test_contract_on_the_cut_family():
    with cut_env(2, max_episode_steps=3, strict=False, evaluation_stream={"every": 2}) as env:
        report = check_vector_env_contract(env, steps=3, reset_ids=(1,), reset_step=1)
        receipt = env.receipt()
    assert report["status"] == "PASS" and report["evaluation_mode"] == STREAM_VERSION
    assert receipt["reward"]["version"] == SEVER and receipt["reward"]["status"] == "PROVISIONAL_SHAPING_NOT_SUCCESS"
    from robotic_cutting_sim.rl.reward_sever import bonds_for_task
    _, bonds = bonds_for_task(compile_workload_task(particle_spacing_m=SPACING))
    assert receipt["reward"]["target_bonds"] == [len(bonds.target_a)] * 2 and len(bonds.target_a) > 0
    fidelity = receipt["fidelity"]
    assert fidelity["reference"]["id"] == NATIVE_CUT_V7_2_REFERENCE["id"] and fidelity["reduced_fidelity"]
    assert any(r.startswith("particle_spacing_m=") for r in fidelity["reasons"])
    assert any(r.startswith("episode_steps=3") for r in fidelity["reasons"])
    assert "no_finger_pad_obstacles" in fidelity["reasons"]
    assert receipt["task_family"]["kind"] == "ConfiguredTaskFamily" and receipt["task_family"]["workload_id"] == "cut_v7_2_v1"
    assert [t["scene_seed"] for t in receipt["tasks"]] == [None, None]
    assert all(r["verdict"] != "PASS" for r in report["evaluation"])       # 3 steps from HOME: nothing is cut


def test_declared_episode_and_reward_dispatch():
    task = compile_workload_task(particle_spacing_m=SPACING)
    with cut_env(1) as env:
        assert env.declared_episode_steps == env.max_episode_steps == 700
        assert env.reward_version == SEVER and env.tasks[0].task_id == task.task_id
        assert env.material.packed.uniform["max_cuts"] == 4
        r = env.receipt()["reward"]
        assert r["off_target_bonds"] == [r["per_env"][0]["off_target_bonds"]]
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
        with cut_env(1, reward=VE.REWARD, max_episode_steps=2):
            pass
    with cut_env(1, reward=VE.REWARD_BY_GOAL, max_episode_steps=2) as env:
        assert env.reward_version == SEVER


def test_refusals():
    with pytest.raises(UnsupportedBatchConfiguration, match="^UNSUPPORTED_SOLVER"):
        ConfiguredTaskFamily.uniform(str(ROOT / "configs/wp12/scenes/bar_flat_board.json"),
                                     str(ROOT / "configs/wp14/goals/standard_cut.json"), 2)
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    from robotic_cutting_sim.rl.cut_workload import workload_task_args
    args = workload_task_args()
    mixed = ConfiguredTaskFamily(scenes=(args["scene"], generate_scene(512)), goals=(args["goal"], goal_config()),
                                 planner_options=(args["planner_options"], None), particle_spacing_m=SPACING,
                                 max_episode_steps=2)
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_NONUNIFORM_BATCH"):
        open_vector_env(2, task=mixed, device="cpu", parity_mode=True).__enter__()
    with pytest.raises(ValueError):
        ConfiguredTaskFamily(scenes=(args["scene"],), goals=())
    with cut_env(2, max_episode_steps=2) as env:
        with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_GEOMETRY_CHANGE_ON_PARTIAL_RESET"):
            env.reset(env_ids=[0], scene_seeds=[512])
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_TASK"):
        VE.resolve_tasks(2, cut_family(2, max_episode_steps=2), [512, 15])


def _single(task, pose, seed=42):
    config = CouplingConfig(control_dt=task.control_dt_s, material_dt=1e-4, dx=task.grid_dx_m, device="cpu",
                            feedback=True)
    m = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                            solver="tlmpm_v7_2", v7_2_optimizations=True)
    m.reset(pose, seed)
    return m


def _lockstep(task, envs, start_step, steps):
    """Drive vector envs (each a list of its env indices) with reference_command_policy_v1 actions from knot
    ``start_step`` and a frozen single solver with the env's commanded poses; bitwise comparison every step."""
    actions, _, schedule, report = reference_command_actions(task, steps, start_step)
    assert report["max_position_diff_m"] <= 1e-9
    single = _single(task, np.asarray(schedule["initial_pose"]))
    for env in envs:
        env.reset(seeds=[42] * env.num_envs)
        for e in range(env.num_envs):
            assert compare_env(env.material.solver, e, single.bed) == [], ("reset", e)
    policies = [ReferenceCommandPolicyV1([actions], [0] * env.num_envs, device="cpu") for env in envs]
    infos = [{"step_count": torch.zeros(env.num_envs, dtype=torch.int64)} for env in envs]
    rewards = None
    for k in range(steps):
        out = [env.step(policy(None, info)) for env, policy, info in zip(envs, policies, infos)]
        infos = [o[4] for o in out]
        pose = envs[0].commanded_pose[0].numpy().copy()
        cut = int(infos[0]["cut_digit"][0])
        assert cut == int(schedule["resolved_cut"][k])
        single.advance(pose, (k + 1) * task.control_dt_s, cutting=cut >= 0, cut_index=max(cut, 0))
        for env, o in zip(envs, out):
            for e in range(env.num_envs):
                assert np.array_equal(env.commanded_pose[e].numpy(), pose)
                assert compare_env(env.material.solver, e, single.bed) == [], (k, env.num_envs, e)
            assert torch.equal(o[1], out[0][1][:1].repeat(env.num_envs))
        rewards = out
    return single, infos


def test_cpu_l0_cut_n1_bitwise_through_contact_labels_and_link_completion():
    """Batched N=1 equals the frozen single tlmpm_v7_2 bitwise from the first armed knot through blade contact,
    the first labels and finite-link completion events (2.5 mm CUT, reference_command_policy_v1 actions)."""
    task = compile_workload_task(particle_spacing_m=SPACING)
    first_armed = int(np.argmax(np.asarray(task.reference_cut_indices) >= 0))
    start = tuple(task.reference_edge_pose_world(first_armed * task.control_dt_s))
    with open_vector_env(1, task=cut_family(1, initial_edge_poses=(start,)), device="cpu", parity_mode=True) as env:
        single, infos = _lockstep(task, [env], first_armed, LINK_WINDOW_STEPS)
        codes = single.history_state()["codes"]
        links = single.bed.solver_receipt()["link_completion_counts"]["finite_link_events"]
        assert np.count_nonzero(codes) > 0 and links > 0, (np.count_nonzero(codes), links)
        assert int(infos[0]["severed_target_bonds"][0]) > 0


def test_cpu_l0_cut_n3_copies_bitwise_and_partial_reset():
    """Every env of N=3 identical copies equals the frozen single bitwise over a window with blade contact and
    labelling; then env 1 is partially reset and one more step is taken: env 1 equals a freshly reset single
    stepped once with its input, envs 0 and 2 equal the continuing single (non-reset envs untouched)."""
    task = compile_workload_task(particle_spacing_m=SPACING)
    start = tuple(task.reference_edge_pose_world(WINDOW_START * task.control_dt_s))
    with open_vector_env(3, task=cut_family(3, initial_edge_poses=(start,) * 3), device="cpu",
                         parity_mode=True) as env:
        single, _ = _lockstep(task, [env], WINDOW_START, WINDOW_STEPS)
        assert np.count_nonzero(single.history_state()["codes"]) > 0
        actions, _, _, _ = reference_command_actions(task, WINDOW_STEPS + 1, WINDOW_START)
        obs, info = env.reset(env_ids=[1], seeds=[42])
        assert info["step_count"].tolist() == [WINDOW_STEPS, 0, WINDOW_STEPS]
        a = torch.as_tensor(np.stack([actions[WINDOW_STEPS], actions[0], actions[WINDOW_STEPS]]))
        env.step(a)
        pose0, pose1 = env.commanded_pose[0].numpy().copy(), env.commanded_pose[1].numpy().copy()
        assert np.array_equal(env.commanded_pose[2].numpy(), pose0)
        cut = int(np.asarray(task.reference_cut_indices)[WINDOW_START + WINDOW_STEPS])
        single.advance(pose0, (WINDOW_STEPS + 1) * task.control_dt_s, cutting=cut >= 0, cut_index=max(cut, 0))
        for e in (0, 2):
            assert compare_env(env.material.solver, e, single.bed) == [], f"env {e} changed by the reset of env 1"
        fresh = _single(task, np.asarray(start))
        cut1 = int(np.asarray(task.reference_cut_indices)[WINDOW_START])
        fresh.advance(pose1, task.control_dt_s, cutting=cut1 >= 0, cut_index=max(cut1, 0))
        assert compare_env(env.material.solver, 1, fresh.bed) == [], "reset env differs from a fresh single"


def _state(env, e):
    state = env.material.snapshot(e)
    for name in ("pending_wrench", "applied_previous", "step_count", "seed", "last_pose", "invalid_reason"):
        state[f"material.{name}"] = getattr(env.material, name).numpy()[e].copy()
    state["reward_counts"] = env.reward.counts[:, e].numpy().copy()
    return state


def test_measured_replay_drive_with_pads_on_the_cut_family():
    """The WP-12 replay adapter's trajectory drives a configured CUT env through step_measured_edge with the
    handle + two measured pads (T5.5 driver in the vector env)."""
    from robotic_cutting_sim.rl.policies_batched import MeasuredTrajectoryDriver
    from robotic_cutting_sim.rl.replay_wp12 import extract_wp12_trajectory
    episode = ROOT / "artifacts" / "wp12" / "demos_001" / "DEMO1_A" / "run" / "episode_000"
    if not (episode / "actions.jsonl.gz").exists():
        pytest.skip("WP-12 Demo 1 recording not present (NOT_RUN)")
    traj = extract_wp12_trajectory(episode)
    pads = (traj.reset_pad_boxes_edge.tolist(),) * 2
    with cut_env(2, max_episode_steps=3, initial_edge_poses=(tuple(traj.reset_edge_pose_world_xyzw),) * 2,
                 pad_boxes_edge=pads, pad_recipients=traj.pad_recipients) as env:
        driver = MeasuredTrajectoryDriver([traj], [0, 0], device="cpu")
        _, info = env.reset(seeds=[42, 42])
        for _ in range(2):
            pose, cut, boxes = driver.inputs(info["step_count"])
            _, _, _, _, info = env.step_measured_edge(pose, cut, pad_boxes=boxes)
            assert info["drive"].tolist() == [1, 1]
        assert np.array_equal(env.commanded_pose.numpy()[0], traj.edge_pose_world_xyzw[1])
        per_env = env.receipt()["fidelity"]["per_env"]
        assert [p["replay_steps_with_measured_pads"] for p in per_env] == [2, 2]
        assert env.boxes == 3

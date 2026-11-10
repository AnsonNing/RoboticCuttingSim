"""P1-2b vector-env additions on the Warp/Torch CPU device (workload A): ``tasks=``, ``termination_reason``,
infrastructure-error classification, the material input guard, the evaluation stream, and the workload B
dispatch refusals that need no Isaac Lab.

Torch test: run in ``.deps/isaacsim-venv`` with the D-09 overlay; in ``.venv`` (no torch) it is skipped = NOT_RUN.
Coarse, shortened TRIM tasks (2.5 mm spacing, a few steps), as ``test_vector_env_api.py``: every env here is
``reduced_fidelity``. CPU kernels run in task order, so the isolation checks are bitwise."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import warp as wp  # noqa: E402

from test_vector_env_api import differing, env_of, env_state, family, push  # noqa: E402
from robotic_cutting_sim.batched.material_batch import InvalidBatchEnvs  # noqa: E402
from robotic_cutting_sim.rl import TaskFamily, UnsupportedVectorConfiguration, open_vector_env  # noqa: E402
from robotic_cutting_sim.rl import termination as T  # noqa: E402
from robotic_cutting_sim.rl import vector_env as VE  # noqa: E402
from robotic_cutting_sim.rl.evaluation_stream import DEFAULT_MODE, STREAM_VERSION  # noqa: E402
from robotic_cutting_sim.rl.material_guard import (NONFINITE_POSE, OUT_OF_DOMAIN, PAD_BOX,  # noqa: E402
                                                   TRANSLATION_JUMP, MaterialInputGuard)
from robotic_cutting_sim.rl.vector_contract import check_vector_env_contract  # noqa: E402

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]
ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- contract (workload A)

@pytest.mark.parametrize("profile,stream", [("state", True), ("privileged_state", False)])
def test_vector_contract_workload_a(profile, stream):
    with env_of(observation_profile=profile, strict=False,
                evaluation_stream={"every": 2} if stream else None) as env:
        report = check_vector_env_contract(env, steps=3, reset_ids=(1,), reset_step=1)
    assert report["status"] == "PASS" and report["backend"] == "kinematic_ideal_tracking_batched"
    assert report["evaluation_mode"] == (STREAM_VERSION if stream else DEFAULT_MODE)
    assert report["final_step_count"] == [3, 1]
    assert report["termination_reason"] == [["TRUNCATED_HORIZON"], []]


# ----------------------------------------------------------------------------- tasks=

def test_tasks_selection_heterogeneous_batch_and_refusals():
    template = family((512, 15))
    with open_vector_env(2, task=template, tasks=[512, {"scene_seed": 15, "seed": 9}], device="cpu",
                         parity_mode=True) as env:
        assert env.task_family.scene_seeds == (512, 15) and env.task_family.env_seeds == (42, 9)
        tasks = env.receipt()["tasks"]
        assert [t["scene_seed"] for t in tasks] == [512, 15] and tasks[0]["config_sha256"] != tasks[1]["config_sha256"]
        assert [t["initial_seed"] for t in tasks] == [42, 9]
        _, info = env.reset()
        assert info["seed"].tolist() == [42, 9]
    plain = TaskFamily(scene_seeds=(0, 0), particle_spacing_m=.0025, max_episode_steps=2)
    resolved = VE.resolve_tasks(2, plain, [512, {"scene_seed": 7, "family_id": "agentic_trim_ood"}])
    assert resolved.family_ids == ("agentic_trim_id", "agentic_trim_ood") and resolved.env_seeds is None
    with open_vector_env(2, task=plain, tasks=[512, {"scene_seed": 7, "family_id": "agentic_trim_ood"}],
                         device="cpu", parity_mode=True) as env:
        assert [t["family_id"] for t in env.receipt()["tasks"]] == ["agentic_trim_id", "agentic_trim_ood"]
    for bad in ([object(), 15], [{"scene_seed": 1, "geometry": "x"}, 15], [{"seed": 3}, 15],
                [{"scene_seed": 1, "family_id": "cut_01"}, 15]):
        with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_TASK"):
            VE.resolve_tasks(2, None, bad)
    with pytest.raises(ValueError, match="one entry per env"):
        VE.resolve_tasks(3, None, [512, 15])


# ----------------------------------------------------------------------------- termination_reason

def test_termination_reason_material_invalid_and_truncation_bits():
    with env_of(strict=False) as env:
        env.reset()
        cut = torch.tensor([5, -1], dtype=torch.int32)     # digit 5 >= K=2: material INVALID_CUT_DIGIT (env 0)
        _, reward, terminated, truncated, info = env.step_measured_edge(env.commanded_pose.clone(), cut)
        assert info["termination_reason"].tolist() == [T.TERM_MATERIAL_INVALID, 0]
        assert terminated.tolist() == [True, False] and truncated.tolist() == [False, False]
        assert bool(info["invalid_penalty_applied"][0]) and not bool(info["invalid_penalty_applied"][1])
        for _ in range(2):
            _, _, terminated, truncated, info = env.step(push(2))
        assert truncated.tolist() == [True, True] and terminated.tolist() == [True, False]
        assert [T.decode_termination(v) for v in info["termination_reason"].tolist()] == [
            ["MATERIAL_INVALID", "TRUNCATED_HORIZON"], ["TRUNCATED_HORIZON"]]
        assert env.receipt()["termination"]["bits"]["64"] == "INFRASTRUCTURE_ERROR"
        _, info = env.reset(env_ids=[0])
        _, _, terminated, _, info = env.step(push(2))
        assert int(info["termination_reason"][0]) == 0 and not bool(terminated[0])


def test_infrastructure_error_is_not_a_task_failure(monkeypatch):
    with env_of(seeds=(512,), strict=True) as env:
        env.reset()
        with pytest.raises(InvalidBatchEnvs):       # task failure: strict-mode invalid env, propagated unchanged
            env.step_measured_edge(env.commanded_pose.clone(), torch.tensor([5], dtype=torch.int32))
        assert env.infrastructure_error is None
    with env_of(seeds=(512, 15), strict=False) as env:
        env.reset()

        def broken(*a, **k):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        monkeypatch.setattr(env.material, "advance", broken)
        with pytest.raises(T.VectorEnvInfrastructureError) as error:
            env.step(push(2))
        assert error.value.kind == "CUDA_ERROR" and str(error.value).startswith("INFRASTRUCTURE_ERROR[CUDA_ERROR]")
        assert error.value.termination_reason.tolist() == [T.TERM_INFRASTRUCTURE] * 2
        assert T.classify_exception(error.value) == ("infrastructure", "CUDA_ERROR")
        assert env.receipt()["infrastructure_error"]["kind"] == "CUDA_ERROR"
        with pytest.raises(RuntimeError, match="infrastructure error"):
            env.step(push(2))
    assert T.classify_exception(torch.cuda.OutOfMemoryError("CUDA out of memory")) == ("infrastructure", "OOM")
    assert T.classify_exception(ValueError("actions must be a torch tensor"))[0] == "caller"


# ----------------------------------------------------------------------------- material input guard (task 5)

def test_input_guard_holds_out_of_domain_envs_and_leaves_the_others_bitwise():
    seeds = (512, 15, 512)
    with env_of(seeds, strict=False) as run, env_of(seeds, strict=False) as reference:
        run.reset()
        reference.reset()
        a = push(3)
        run.step(a)
        reference.step(a)
        held_pose = run.material.last_pose.numpy()[1:].copy()
        edge = run.commanded_pose.clone()
        edge[1, 2] += 1.0                       # 1 m above the workpiece: OUT_OF_DOMAIN + TRANSLATION_JUMP
        edge[2, :] = float("nan")               # NONFINITE_POSE
        cut = torch.zeros(3, dtype=torch.int32)
        mask = torch.tensor([False, True, True])
        _, reward, terminated, _, info = run.step_measured_edge(edge, cut, env_mask=mask, actions=a)
        reference.step(a)
        assert differing(env_state(run, 0), env_state(reference, 0)) == [], "env 0 changed by the held envs"
        reasons = info["input_guard_reason"].tolist()
        assert reasons[0] == 0 and reasons[1] & OUT_OF_DOMAIN and reasons[1] & TRANSLATION_JUMP
        assert reasons[2] == NONFINITE_POSE
        assert info["termination_reason"].tolist() == [0, T.TERM_INPUT_DOMAIN, T.TERM_INPUT_DOMAIN]
        assert terminated.tolist() == [False, True, True] and bool(torch.isfinite(reward).all())
        # the material never saw the bad input: held at the last accepted pose, disarmed, no material flag
        assert np.array_equal(run.material.last_pose.numpy()[1:], held_pose)
        assert run.material.cut_input.numpy()[1:].tolist() == [-1, -1]
        assert run.material.invalid_reason.numpy().tolist() == [0, 0, 0]
        assert np.isfinite(run.material.solver.x.numpy()).all()
        # sticky until reset: a legitimate action keeps envs 1 and 2 held
        run.step(a)
        reference.step(a)
        assert differing(env_state(run, 0), env_state(reference, 0)) == []
        assert np.array_equal(run.material.last_pose.numpy()[1:], held_pose)
        assert int(run.input_guard.trip_step[1]) == 2 and run.input_guard.trips.tolist() == [0, 1, 1]
        _, info = run.reset(env_ids=[1])
        assert run.input_guard.reason.tolist() == [0, 0, NONFINITE_POSE]
        _, _, terminated, _, info = run.step(a)
        assert int(info["termination_reason"][1]) == 0 and terminated.tolist() == [False, False, True]
    with env_of(seeds=(512,), input_guard=False) as off:     # rollback: the P1-2 behaviour (no guard)
        assert off.input_guard is None and off.receipt()["input_guard"]["enabled"] is False


def test_input_guard_pad_boxes_of_both_backends():
    """Workload A replay pads (written only for accepted envs) and the workload B in-place path
    (``apply_in_place_boxes``: the pad kernel wrote the boxes; held envs get their last accepted boxes back)."""
    from robotic_cutting_sim.tool_profile import handle_box_edge
    seeds = (512, 15)
    tasks = [VE._compile(family(seeds), s) for s in seeds]
    handle = np.asarray(handle_box_edge(tasks[0].tool_profile), np.float32)
    pad = np.array([[0., .05, .02], [.008, .008, .008]], np.float32)
    boxes = np.stack([np.stack([handle, pad, pad])] * 2).astype(np.float32)
    with env_of(seeds, strict=False, pad_boxes_edge=boxes.tolist(), pad_recipients=(None, "left", "right")) as env:
        env.reset()
        new = torch.as_tensor(boxes.copy())
        new[1, 2, 0, 1] = 2.0                  # env 1: a finger pad 2 m from the edge
        before = env._box_centers[1].clone()
        _, _, terminated, _, info = env.step_measured_edge(env.commanded_pose.clone(),
                                                           torch.zeros(2, dtype=torch.int32), pad_boxes=new)
        assert info["input_guard_reason"].tolist() == [0, PAD_BOX] and terminated.tolist() == [False, True]
        assert torch.equal(env._box_centers[0], new[0, :, 0]) and torch.equal(env._box_centers[1], before)
        # workload B path on the same material: boxes are already written in place
        guard = MaterialInputGuard(env.material, 2, env.torch_device)
        centers, halves = env._box_centers, env._box_halves
        guard.init_boxes(centers, halves)
        accepted = centers.clone()
        centers[0, 1, 0] = float("nan")
        pose, cut = guard.apply_in_place_boxes(env._last_pose.clone(), torch.zeros(2, dtype=torch.int32),
                                               env._last_pose, centers, halves)
        assert guard.reason.tolist() == [PAD_BOX, 0]
        assert torch.equal(centers[0], accepted[0]) and cut.tolist() == [-1, 0]
        assert torch.equal(pose, env._last_pose)


# ----------------------------------------------------------------------------- evaluation stream (task 4)

def test_evaluation_stream_records_the_native_cadence_and_feeds_the_frozen_evaluator():
    from robotic_cutting_sim.goals import evaluate_goal
    with env_of(steps=5, strict=False, evaluation_stream={"every": 2}) as env, env_of(steps=5, strict=False) as plain:
        env.reset()
        plain.reset()
        snaps = {0: [env.snapshot_world(e) for e in range(2)]}
        for k in range(1, 6):
            if k == 4:
                env.reset(env_ids=[1])
                plain.reset(env_ids=[1])
            env.step(push(2))
            plain.step(push(2))
            snaps[k] = [env.snapshot_world(e) for e in range(2)]
        assert env.evaluation_stream.count.tolist() == [4, 2]     # env 0: 0, 2, 4, 5 (done); env 1: reset, 2
        frames, dropped = env.evaluation_stream.host_frames(0, env._env_host[0]["T_world_local"], env.control_dt)
        assert [f["step_count"] for f in frames] == [0, 2, 4, 5] and dropped == 0
        for f, k in zip(frames, (0, 2, 4, 5)):
            s = snaps[k][0]
            assert np.array_equal(f["positions"], s["positions"]) and np.array_equal(f["codes"], s["codes"])
            assert np.array_equal(f["velocities"], s["velocities"])
        results = env.evaluate()
        assert [r["frames"] for r in results] == [4, 2] and all(r["evaluation_mode"] == STREAM_VERSION for r in results)
        assert results[0]["reference_frame_at_t0"] is False and results[0]["stream_samples"] == 4
        task, ref = env.tasks[0], env.evaluation_reference(0)
        data = dict(ref, time_s=np.array([f["time_s"] for f in frames]),
                    positions=np.stack([f["positions"] for f in frames]),
                    velocities=np.stack([f["velocities"] for f in frames]),
                    codes=np.stack([f["codes"].astype(np.int32) for f in frames]),
                    code_encoding="packed2", max_cuts=2, particle_spacing_m=task.particle_spacing_m)
        direct = evaluate_goal(task.goal, data, task=task)
        assert direct["verdict"] == results[0]["result"]["verdict"]
        assert direct["integrity"] == results[0]["result"]["integrity"]
        assert direct["outcome"] == results[0]["result"]["outcome"]
        default = plain.evaluate()
        assert [r["frames"] for r in default] == [2, 2] and all(r["evaluation_mode"] == DEFAULT_MODE for r in default)
        assert env.receipt()["evaluation_mode"] == STREAM_VERSION and plain.receipt()["evaluation_mode"] == DEFAULT_MODE
        with pytest.raises(ValueError, match="exclusive"):
            env.evaluate([0], frames={0: [snaps[2][0]]})
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
        env_of(evaluation_stream=True, autoreset="same_step").__enter__()


def test_evaluation_stream_full_episode_has_the_native_121_samples():
    fam = TaskFamily(scene_seeds=(512,), particle_spacing_m=.0025)
    with open_vector_env(1, task=fam, device="cpu", parity_mode=True, evaluation_stream=True) as env:
        stream = env.evaluation_stream
        assert env.max_episode_steps == 1200 and stream.capacity == 122
        steps = torch.arange(0, 1201)
        due = stream.due(steps, steps >= env.max_episode_steps)
        times = (steps[due].double() * env.control_dt).numpy()
        assert len(times) == 121 and times[0] == 0. and np.isclose(times[-1], 24.)
        native = ROOT / "artifacts/wp17/native_optimized_001/seed_512/box/env/recording/attempt_000"
        if (native / "physics_stream.json").exists():
            checkpoint = json.loads((native / "physics_stream.json").read_text())
            native_t = np.fromfile(native / "physics_time_s.bin", dtype="<f8")[:checkpoint["sample_count"]]
            assert checkpoint["sample_count"] == 121 and np.allclose(native_t, times, rtol=0, atol=1e-9)
        assert env.receipt()["evaluation_stream"]["capacity_samples_per_env"] == 122


def test_stream_and_guard_add_no_host_sync(monkeypatch):
    counts = {"numpy": 0, "torch": 0}
    original = wp.array.numpy

    def counted(self):
        counts["numpy"] += 1
        return original(self)

    for strict, expected in ((True, 1), (False, 0)):
        with env_of(seeds=(512, 15), strict=strict, evaluation_stream={"every": 1}) as env:
            env.reset()
            monkeypatch.setattr(wp.array, "numpy", counted)
            for name in ("item", "tolist", "cpu", "numpy", "nonzero"):
                orig = getattr(torch.Tensor, name)
                monkeypatch.setattr(torch.Tensor, name, (lambda o: lambda self, *a, **k: (
                    counts.__setitem__("torch", counts["torch"] + 1), o(self, *a, **k))[1])(orig))
            counts.update(numpy=0, torch=0)
            env.step(push(2))
            observed = dict(counts)
            monkeypatch.undo()
        assert observed == {"numpy": expected, "torch": 0}, (strict, observed)


# ----------------------------------------------------------------------------- workload B dispatch refusals

def test_isaac_lab_newton_dispatch_refusals_without_isaac_lab():
    fam = TaskFamily(scene_seeds=(512, 15), particle_spacing_m=.0025, max_episode_steps=3)
    cases = [(dict(cameras=["wrist_d405_0"]), "UNSUPPORTED_SENSOR"),
             (dict(observation_profile="vision_proprio"), "UNSUPPORTED_SENSOR"),
             (dict(autoreset="same_step"), "UNSUPPORTED_AUTORESET_FINAL_OBS"),
             (dict(autoreset="next_step"), "UNSUPPORTED_OPTION"),
             (dict(device="cpu"), "UNSUPPORTED_RIGID_CPU"),
             (dict(material="rubber_material_v1"), "UNSUPPORTED_OPTION"),
             (dict(recorders=["cut_events"]), "UNSUPPORTED_RECORDER"),
             (dict(debug_sync=True), "UNSUPPORTED_OPTION"),
             (dict(runtime="isaac_sim"), "UNSUPPORTED_RUNTIME")]
    for options, code in cases:
        kwargs = dict(device="cuda", **options) if "device" not in options else options
        with pytest.raises(UnsupportedVectorConfiguration, match=f"^{code}"):
            open_vector_env(2, task=fam, backend="isaac_lab_newton", **kwargs).__enter__()
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
        open_vector_env(2, task=family((512, 15)), backend="isaac_lab_newton", device="cpu",
                        cpu_test_mode=True).__enter__()           # initial_edge_poses are workload A options
    try:
        import isaaclab  # noqa: F401
    except ImportError:
        with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_BACKEND"):
            open_vector_env(2, task=fam, backend="isaac_lab_newton", device="cuda").__enter__()


# ----------------------------------------------------------------------------- review F5 / F7

def test_autoreset_never_scores_the_fresh_state_and_terminal_capture_scores_the_finished_episode():
    """Review F5 (``autoreset_evaluate_probe.py``): after a same-step autoreset, ``evaluate()`` used to score the fresh
    state silently (1 frame, step 0, FAIL_INTEGRITY). Now it refuses, and ``terminal_capture=True`` scores the
    finished episode (state captured before the reset, with the stream's samples of that episode)."""
    with env_of(seeds=(512,), autoreset="same_step") as env:
        env.reset()
        for _ in range(STEPS_SHORT):
            _, _, _, truncated, info = env.step(push(1))
        assert bool(info["autoreset"][0]) and "terminal_captured" not in info
        with pytest.raises(RuntimeError, match="^NOT_EVALUATED"):
            env.evaluate([0])
        with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
            env.evaluate([0], episode="last_completed")
        assert env.receipt()["terminal_capture"]["enabled"] is False
    with env_of(seeds=(512,), autoreset="same_step", terminal_capture=True, evaluation_stream={"every": 1}) as env, \
            env_of(seeds=(512,)) as plain:
        env.reset()
        plain.reset()
        with pytest.raises(RuntimeError, match="^NOT_EVALUATED"):
            env.evaluate([0], episode="last_completed")      # nothing finished yet
        for _ in range(STEPS_SHORT):
            _, _, _, _, info = env.step(push(1))
            plain.step(push(1))
        assert info["terminal_captured"].tolist() == [True]
        expected = plain.evaluate([0])[0]
        done = env.evaluate([0], episode="last_completed")[0]
        assert done["episode_scored"] == "last_completed" and done["step_count"] == STEPS_SHORT
        assert done["final_codes_sha256"] == expected["final_codes_sha256"] and done["labelled"] > 0
        assert done["frames"] == STEPS_SHORT + 1 and done["evaluation_mode"] == STREAM_VERSION
        assert done["summary"]["outcome"] == expected["summary"]["outcome"]
        assert done["completed_episode_index"] == 1
        env.step(push(1))                                     # the next episode keeps its own samples
        current = env.evaluate([0])[0]
        assert current["step_count"] == 1 and current["frames"] == 2
        again = env.evaluate([0], episode="last_completed")[0]
        assert again["final_codes_sha256"] == done["final_codes_sha256"] and again["frames"] == done["frames"]
        assert env.receipt()["terminal_capture"]["enabled"] is True
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
        env_of(seeds=(512,), terminal_capture=True).__enter__()           # nothing to capture without autoreset


STEPS_SHORT = 3


def test_multiple_device_hashgrids_is_an_explicit_diagnostics_opt_in():
    """Review F7: the vector env forwards ``allow_multiple_device_hashgrids`` (refused by default) and records it."""
    with env_of(seeds=(512,)) as env:
        receipt = env.receipt()
        assert receipt["allow_multiple_device_hashgrids"] is False
        assert receipt["device_hashgrid_guard"] == "not_applicable_host_grid"     # CPU grids carry no descriptor
        assert env.material.allow_multiple_device_hashgrids is False
    with env_of(seeds=(512,), allow_multiple_device_hashgrids=True) as env:
        receipt = env.receipt()
        assert receipt["allow_multiple_device_hashgrids"] is True and receipt["device_hashgrid_guard"] == "disabled"
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_OPTION"):
        open_vector_env(2, task=TaskFamily(scene_seeds=(512, 15)), backend="isaac_lab_newton", device="cuda",
                        allow_multiple_device_hashgrids=True).__enter__()

"""P1-4 T4.1: ledger, classification, info adapter, schedule and supervisor fault injection (CPU, no Torch).

The end-to-end tests run the real ``Supervisor`` over the real child script ``run_long_rollout.py`` with its
NumPy ``fake`` backend (a fake child), injecting faults through ``RCS_FAULT_INJECT``."""
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "rl_ready"))

from robotic_cutting_sim.rl import supervision as S  # noqa: E402
from robotic_cutting_sim.rl.long_rollout import RolloutSchedule  # noqa: E402

CHILD = ROOT / "scripts" / "rl_ready" / "run_long_rollout.py"
CONFIG = ROOT / "configs" / "rl_benchmark" / "long_rollout_v1.json"


# ----------------------------------------------------------------------------- ledger

def test_ledger_append_only_fsync_and_exclusive(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(S.os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    path = tmp_path / "ledger.jsonl"
    with S.Ledger(path) as ledger:
        ledger.write("start", a=1)
        ledger.write("heartbeat", global_step=5, value=np.float32(1.5), arr=np.arange(3))
    assert len(calls) == 2                                   # one fsync per record
    records, report = S.read_ledger(path)
    assert [r["type"] for r in records] == ["start", "heartbeat"]
    assert records[0]["schema"] == S.LEDGER_SCHEMA and records[1]["seq"] == 1 and records[1]["arr"] == [0, 1, 2]
    assert not report["truncated_tail"]
    with pytest.raises(FileExistsError):                     # never reopened / overwritten
        S.Ledger(path)


def test_read_ledger_tolerates_truncated_tail_and_corrupt_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"type": "start", "seq": 0}\nnot json\n{"type": "heartbeat", "se')
    records, report = S.read_ledger(path)
    assert [r["type"] for r in records] == ["start"]
    assert report["truncated_tail"] and not report["truncated_tail_parsed"] and report["corrupt_lines"] == 1
    path.write_bytes(b'{"type": "start", "seq": 0}\n{"type": "end", "seq": 1}')     # complete record, no newline
    records, report = S.read_ledger(path)
    assert [r["type"] for r in records] == ["start", "end"] and report["truncated_tail_parsed"]
    assert S.read_ledger(tmp_path / "missing.jsonl") == ([], dict(path=str(tmp_path / "missing.jsonl"), exists=False,
                                                                  truncated_tail=False, corrupt_lines=0, records=0))


def test_ledger_follower_reads_complete_lines_only(tmp_path):
    path = tmp_path / "l.jsonl"
    follower = S.LedgerFollower(path)
    assert follower.poll() == []
    path.write_bytes(b'{"type": "a"}\n{"type": "b"')
    assert [r["type"] for r in follower.poll()] == ["a"]
    with path.open("ab") as f:
        f.write(b'}\n{"type": "c"}\n')
    assert [r["type"] for r in follower.poll()] == ["b", "c"]


# ----------------------------------------------------------------------------- classification

@pytest.mark.parametrize("exc_type,message,expected", [
    ("MemoryError", "", "INFRA_OOM_HOST"),
    ("RuntimeError", "std::bad_alloc", "INFRA_OOM_HOST"),
    ("OutOfMemoryError", "CUDA out of memory. Tried to allocate 2 GiB", "INFRA_OOM_DEVICE"),
    ("RuntimeError", "CUDA out of memory. Tried to allocate 20.00 GiB", "INFRA_OOM_DEVICE"),
    ("RuntimeError", "Warp CUDA error 2: out of memory", "INFRA_OOM_DEVICE"),
    ("RuntimeError", "CUDA error: an illegal memory access was encountered", "INFRA_CUDA_ERROR"),
    ("RuntimeError", "Warp CUDA error 719: unspecified launch failure", "INFRA_CUDA_ERROR"),
    ("OSError", "[Errno 28] No space left on device", "INFRA_DISK"),
    ("FloatingPointError", "Invalid batched envs [1]: {1: ['NONFINITE_STATE']}", "INFRA_INVALID_STATE"),
    ("InvalidBatchEnvs", "x", "INFRA_INVALID_STATE"),
    ("KeyError", "'foo'", None),
])
def test_classify_error_text(exc_type, message, expected):
    assert S.classify_error_text(exc_type, message) == expected


def _unverified(pid):
    return dict(status="UNVERIFIED")


@pytest.mark.parametrize("returncode,records,stderr,reason,expected", [
    (0, [{"type": "end"}], "", None, "COMPLETE"),
    (0, [], "", None, "INFRA_UNKNOWN_EXIT"),
    (-11, [], "Fatal Python error: Segmentation fault", None, "INFRA_CRASH_SIGNAL"),
    (-6, [], "", None, "INFRA_CRASH_SIGNAL"),
    (-6, [], "Warp CUDA error 700: an illegal memory access was encountered", None, "INFRA_CUDA_ERROR"),
    (-9, [], "", None, "INFRA_KILLED"),
    (137, [], "", None, "INFRA_KILLED"),
    (1, [{"type": "infra_error", "exc_type": "MemoryError", "message": ""}], "", None, "INFRA_OOM_HOST"),
    (1, [{"type": "infra_error", "exc_type": "KeyError", "message": "x"}], "", None, "INFRA_PYTHON_EXCEPTION"),
    (1, [], "Traceback (most recent call last):\n ValueError", None, "INFRA_PYTHON_EXCEPTION"),
    (-15, [{"type": "heartbeat"}], "", "hang", "INFRA_HANG"),
    (-15, [{"type": "heartbeat"}], "", "wall_timeout", "INFRA_TIMEOUT"),
])
def test_classify_process_end(returncode, records, stderr, reason, expected):
    out = S.classify_process_end(returncode, records=records, stderr_tail=stderr, supervisor_reason=reason, pid=123,
                                 oom_probe=_unverified)
    assert out["cls"] == expected
    assert out["infrastructure"] == (expected != "COMPLETE")
    assert out["family"] == S.INFRA_CLASSES.get(expected)
    if expected == "INFRA_KILLED":
        assert out["cause"] == "UNVERIFIED" and out["signal"] == "SIGKILL"


def test_kernel_oom_evidence_uses_readable_logs_only():
    lines = "[1.0] Out of memory: Killed process 4242 (python) total-vm:1kB\n"
    assert S.kernel_oom_evidence(4242, runner=lambda a: lines)["status"] == "VERIFIED_OOM_KILL"
    assert S.kernel_oom_evidence(4243, runner=lambda a: lines)["status"] == "NO_OOM_RECORD"
    assert S.kernel_oom_evidence(4242, runner=lambda a: None)["status"] == "UNVERIFIED"


@pytest.mark.parametrize("kwargs,expected", [
    (dict(cause="full", verdict="PASS"), S.TASK_SUCCESS),
    (dict(cause="full", verdict="FAIL"), S.TASK_FAILURE),
    (dict(cause="full", verdict=None), S.TASK_FAILURE),
    (dict(cause="terminated", termination_reason=2, invalid_reason=64), S.TERMINATED),        # speed guard
    (dict(cause="terminated", termination_reason=2, invalid_reason=4), S.TERMINATED),         # measured pose
    (dict(cause="terminated", termination_reason=2, invalid_reason=1), S.ENV_INVALID_STATE),  # NaN state
    (dict(cause="terminated", termination_reason=2, invalid_reason=2 | 64), S.ENV_INVALID_STATE),
    (dict(cause="terminated", termination_reason=4, infra=1), S.ENV_INVALID_STATE),
    (dict(cause="terminated", termination_reason=4), S.TERMINATED),                           # grasp lost
    (dict(cause="forced"), S.TRUNCATED_BY_FORCED_RESET),
    (dict(cause="run_end"), S.TRUNCATED_RUN_END),
])
def test_classify_episode_end(kwargs, expected):
    assert S.classify_episode_end(**kwargs) == expected


# ----------------------------------------------------------------------------- the info adapter

class _FakeTensor:
    """Torch-like value (``detach().cpu().numpy()``) without Torch."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def test_step_status_legacy_info_derives_material_termination():
    info = {"invalid_reason": _FakeTensor([0, 64, 1, 0]), "step_count": _FakeTensor([3, 3, 3, 1200]),
            "action_status": np.zeros(4)}
    st = S.step_status(info, _FakeTensor([False, True, True, False]), _FakeTensor([False, False, False, True]),
                       num_envs=4)
    assert st.source == "legacy_invalid_reason" and st.infra_key is None
    assert st.termination_reason.tolist() == [0, S.TERM_MATERIAL_INVALID, S.TERM_MATERIAL_INVALID, 0]
    assert st.terminated.tolist() == [False, True, True, False] and st.truncated.tolist() == [False] * 3 + [True]
    assert st.step_count.tolist() == [3, 3, 3, 1200] and "action_status" in st.extra_keys


def test_step_status_prefers_termination_reason_and_reads_infra_key():
    info = {"invalid_reason": np.array([0, 1, 0]), "termination_reason": np.array([4, 2, 0]),
            "infrastructure_error": np.array([0, 1, 0])}
    st = S.step_status(info, None, None, num_envs=3)
    assert st.source == "termination_reason" and st.infra_key == "infrastructure_error"
    assert st.termination_reason.tolist() == [4, 2, 0] and st.infra.tolist() == [0, 1, 0]
    assert st.terminated.tolist() == [True, True, False]          # derived when not given
    assert S.decode_bits(4 | 2, S.TERMINATION_REASONS) == ["MATERIAL_INVALID", "GRASP_LOST"]
    assert S.decode_bits(65, S.INVALID_REASONS) == ["NONFINITE_STATE", "CONTACT_SPEED_GUARD"]


def test_invalid_reason_table_matches_material_batch_source():
    text = (ROOT / "src" / "robotic_cutting_sim" / "batched" / "material_batch.py").read_text()
    for bit, name in S.INVALID_REASONS.items():
        assert f'"{name}"' in text
    assert "INVALID_CONTACT_SPEED_GUARD = 64" in text and "INVALID_NONFINITE_STATE = 1" in text
    lab = (ROOT / "src" / "robotic_cutting_sim" / "rl" / "lab_batched_env.py").read_text()
    assert "TERM_RIGID_INVALID, TERM_MATERIAL_INVALID, TERM_GRASP_LOST, TERM_IK_STREAK = 1, 2, 4, 8" in lab


# ----------------------------------------------------------------------------- seeds and schedule

def test_splitmix64_reference_values_and_episode_seed():
    assert S.splitmix64(0) == 0xE220A8397B1DCDAF        # standard SplitMix64 first output for state 0
    seeds = {S.episode_seed(7, e, i) for e in range(8) for i in range(16)}
    assert len(seeds) == 128 and all(0 <= s < 2 ** 63 for s in seeds)
    assert S.episode_seed(7, 3, 5) == S.episode_seed(7, 3, 5) != S.episode_seed(8, 3, 5)


def _drive(schedule, terminate=lambda step, e: False, limit=100000):
    step, resets = 0, 0
    while not schedule.done() and step < limit:
        step += 1
        ends = schedule.on_step(step, np.array([terminate(step, e) for e in range(schedule.N)]))
        for end in ends:
            schedule.record_end(end["env"], {"full": S.TASK_SUCCESS, "terminated": S.TERMINATED,
                                             "forced": S.TRUNCATED_BY_FORCED_RESET}[end["cause"]],
                                end["episode_index"])
        if ends:
            schedule.begin([end["env"] for end in ends], step)
            resets += len(ends)
    return step, resets


def test_schedule_lr_a_arithmetic_staggered_resets():
    s = RolloutSchedule(8, episode_steps=1200, episodes_per_env=8, offset_steps=150, base_seed=1)
    first = s.initial()
    assert [seg["kind"] for seg in first] == ["episode"] + ["stagger"] * 7
    steps, resets = _drive(s)
    assert steps == s.expected_batch_steps() == 10650
    assert resets == 7 + 8 * 8                       # 7 forced stagger resets + one reset per full episode
    assert s.completed.tolist() == [8] * 8 and s.next_index.min() >= 8


def test_schedule_terminated_episodes_count_and_resume_continues_indices():
    s = RolloutSchedule(2, episode_steps=10, episodes_per_env=3, offset_steps=0, base_seed=5)
    s.initial()
    steps, _ = _drive(s, terminate=lambda step, e: e == 1 and step % 4 == 0)
    assert steps == 30 and s.completed[0] == 3 and s.completed[1] >= 3     # env 1 keeps running beyond quota
    r = RolloutSchedule(2, episode_steps=10, episodes_per_env=3, offset_steps=0, base_seed=5,
                        completed=[2, 1], next_index=[2, 1])
    segs = r.initial()
    assert [seg["episode_index"] for seg in segs] == [2, 1]
    assert segs[0]["seed"] == S.episode_seed(5, 0, 2)


def test_open_segments_and_completed_quota():
    records = [{"type": "start_segments", "global_step": 0, "segments": [dict(env=0, segment=0, episode_index=0),
                                                                          dict(env=1, segment=0, episode_index=-1)]},
               {"type": "episode", "env": 1, "segment": 0, "episode_index": -1, "end_class": S.TRUNCATED_BY_FORCED_RESET},
               {"type": "reset", "global_step": 5, "segments": [dict(env=1, segment=1, episode_index=0)]},
               {"type": "episode", "env": 0, "segment": 0, "episode_index": 0, "end_class": S.TASK_SUCCESS}]
    assert [(s["env"], s["segment"]) for s in S.open_segments(records)] == [(1, 1)]
    counts, nxt = S.completed_quota(records, 2)
    assert counts.tolist() == [1, 0] and nxt.tolist() == [1, 0]


# ----------------------------------------------------------------------------- end to end: supervisor + fake child

def _config(tmp_path, *, num_envs=4, episodes=3, steps=6, offset=2):
    config = json.loads(CONFIG.read_text())
    envs = [dict(env=0, drive="replay_canary", scene_seed=512), dict(env=1, drive="recorded_command_policy_v1",
                                                                      scene_seed=15),
            dict(env=2, drive="random_policy_v1", scene_seed=512), dict(env=3, drive="replay_canary", scene_seed=15)]
    config["runs"]["TEST"] = dict(config["runs"]["LR-A"], num_envs=num_envs, episodes_per_env=episodes,
                                  episode_steps=steps, offset_steps_per_env=offset, envs=envs[:num_envs])
    config["cadence"].update(sample_every_batch_steps=4, heartbeat_every_batch_steps=1)
    config["bounds"]["B2"]["min_samples"] = 3          # short fake runs: few VRAM samples after warm-up
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


def _fake_sampler(pids):
    """Shape of ``rl.telemetry.take_sample`` with one foreign compute process (a co-tenant)."""
    return dict(t=0., ok=True, gpu=dict(index=1, uuid="GPU-fake", name="fake", driver="0", util_pct=40.,
                                        memory_used_mib=12000., memory_total_mib=32607.),
                own_memory_mib=700., foreign_compute={4242: 11000.}, foreign_graphics={}, foreign_memory_mib=11000.,
                display_pids=[], own_sm_pct=None, foreign_sm_pct=None, pmon_available=False)


def _supervise(tmp_path, *, fault=None, step_s=0.02, policy=None, child_args=(), name="run"):
    path, config = _config(tmp_path)

    def command(attempt, attempt_dir, resume_dirs):
        argv = [sys.executable, str(CHILD), "--config", str(path), "--run", "TEST", "--attempt-dir",
                str(attempt_dir), "--backend", "fake", f"--fake-step-s={step_s}", *child_args]
        for d in resume_dirs:
            argv += ["--resume-from", str(d)]
        return argv
    pol = S.SupervisorPolicy(startup_timeout_s=60., hang_min_s=60., hang_factor_median_step=20., term_grace_s=5.,
                             poll_s=.05, sample_interval_s=.02, max_restarts=0, py_spy_on_hang=False)
    for k, v in (policy or {}).items():
        setattr(pol, k, v)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    sup = S.Supervisor(tmp_path / name, command, policy=pol, env=env, fault_inject=fault,
                       sampler=_fake_sampler, oom_probe=_unverified,
                       meta=dict(run="TEST", backend="fake"))
    final = sup.run()
    return final, sup, tmp_path / name, config


def _attempt_records(run_dir, attempt=1):
    return S.read_ledger(run_dir / f"attempt_{attempt:03d}" / "ledger.jsonl")[0]


def test_e2e_complete_run_classified_and_analyzed(tmp_path):
    final, sup, run_dir, config = _supervise(tmp_path)
    assert final["status"] == "COMPLETE" and final["infra_failures"] == {}
    records = _attempt_records(run_dir)
    types = [r["type"] for r in records]
    assert types[0] == "start" and types[-1] == "end" and "infra_error" not in types
    assert next(r for r in records if r["type"] == "adapter")["source"] == "legacy_invalid_reason"
    tele = S.read_ledger(run_dir / "telemetry.jsonl")[0]
    assert tele and all(r["gpu"]["own_memory_mib"] == 700. for r in tele)
    alive = [r for r in tele if r["child"] is not None]          # a last sample may follow the child's exit
    assert alive and all(r["child"]["rss_mib"] > 0 and r["child"]["threads"] >= 1 for r in alive)
    from analyze_long_rollout import analyze
    result = analyze(run_dir, config)
    assert result["acceptance"]["verdict"] == "PASS", result["acceptance"]
    assert result["canaries"]["passed"] == result["canaries"]["episodes"] > 0
    assert result["episodes"]["quota_episodes_per_env"] == {e: n for e, n in
                                                            result["episodes"]["quota_episodes_per_env"].items()
                                                            if n >= 3}
    assert result["infrastructure"]["families"] == {}
    bounds = result["attempts"][0]["bounds"]
    assert bounds["verdict"] == "PASS" and bounds["B2"]["status"] == "PASS" and bounds["B4"]["status"] == "PASS"
    assert result["attempts"][0]["telemetry"]["gpu_sharing"] == "SHARED_GPU"
    # every started segment has exactly one end record
    assert S.open_segments(records) == []


@pytest.mark.parametrize("fault,expected,signal_name", [
    ("sigsegv@5", "INFRA_CRASH_SIGNAL", "SIGSEGV"),
    ("sigabrt@5", "INFRA_CRASH_SIGNAL", "SIGABRT"),
    ("sigkill@5", "INFRA_KILLED", "SIGKILL"),
    ("memoryerror@5", "INFRA_OOM_HOST", None),
    ("cuda_error@5", "INFRA_CUDA_ERROR", None),
    ("cuda_oom@5", "INFRA_OOM_DEVICE", None),
    ("strict_invalid@5", "INFRA_INVALID_STATE", None),
    ("disk@5", "INFRA_DISK", None),
    ("exception@5", "INFRA_PYTHON_EXCEPTION", None),
])
def test_e2e_fault_classes(tmp_path, fault, expected, signal_name):
    final, sup, run_dir, _ = _supervise(tmp_path, fault=fault)
    assert final["status"] == f"FAILED_{expected}"
    assert final["infra_failures"] == {expected: 1}
    cls = sup.attempts[0]["classification"]
    assert cls["cls"] == expected and cls["family"] == S.INFRA_CLASSES[expected]
    if signal_name:
        assert cls["signal"] == signal_name
    records = _attempt_records(run_dir)
    assert any(r["type"] == "fault_injected" for r in records)
    assert not any(r["type"] == "end" for r in records)
    # episodes in flight when the process died are INFRA_INCOMPLETE (supervisor ledger), never task failures
    sup_records = S.read_ledger(run_dir / "supervisor.jsonl")[0]
    incomplete = [r for r in sup_records if r["type"] == "episode_incomplete"]
    assert incomplete and all(r["end_class"] == S.INFRA_INCOMPLETE and r["infra_cls"] == expected
                              for r in incomplete)
    assert {r["env"] for r in incomplete} == {0, 1, 2, 3}


def test_e2e_hang_detected_by_heartbeat_watchdog(tmp_path):
    final, sup, run_dir, _ = _supervise(tmp_path, fault="hang@5", policy=dict(hang_min_s=1.0,
                                                                               hang_factor_median_step=2.))
    assert final["status"] == "FAILED_INFRA_HANG"
    sup_records = S.read_ledger(run_dir / "supervisor.jsonl")[0]
    hang = next(r for r in sup_records if r["type"] == "hang_detected")
    assert hang["step_seen"] and hang["limit_s"] == pytest.approx(1.0)
    kill = next(r for r in sup_records if r["type"] == "watchdog_kill")
    assert kill["sigterm"] and kill["reason"] == "hang"
    assert sup.attempts[0]["classification"]["returncode"] == -signal.SIGTERM


def test_e2e_wall_timeout(tmp_path):
    final, sup, run_dir, _ = _supervise(tmp_path, step_s=0.2, policy=dict(max_wall_s=1.0))
    assert final["status"] == "FAILED_INFRA_TIMEOUT"
    assert sup.attempts[0]["classification"]["family"] == "TIMEOUT"


def test_e2e_nonfinite_env_is_isolated_in_non_strict_mode(tmp_path):
    final, sup, run_dir, config = _supervise(tmp_path, fault="nan_env:2@8")
    assert final["status"] == "COMPLETE"
    eps = [r for r in _attempt_records(run_dir) if r["type"] == "episode"]
    invalid = [r for r in eps if r["end_class"] == S.ENV_INVALID_STATE]
    assert len(invalid) == 1 and invalid[0]["env"] == 2 and invalid[0]["invalid_names"] == ["NONFINITE_STATE"]
    assert invalid[0]["termination_names"] == ["MATERIAL_INVALID"] and invalid[0]["cause"] == "terminated"
    assert invalid[0]["evaluation"] is not None                   # evaluator still runs for diagnostics
    others = [r for r in eps if r["env"] != 2 and r["episode_index"] >= 0 and r["cause"] != "run_end"]
    assert others and all(r["end_class"] == S.TASK_SUCCESS for r in others)
    from analyze_long_rollout import analyze
    result = analyze(run_dir, config)
    assert result["infrastructure"]["env_level_invalid_state"] == 1
    assert result["infrastructure"]["families"] == {"NAN_INVALID_STATE (env level)": 1}
    assert result["episodes"]["task_outcomes_per_env"][2]["failure"] >= 1   # random env fails, invalid not counted


def test_e2e_nonfinite_env_in_strict_mode_is_process_invalid_state(tmp_path):
    final, sup, _, _ = _supervise(tmp_path, fault="nan_env:2@8", child_args=("--fake-strict",))
    assert final["status"] == "FAILED_INFRA_INVALID_STATE"


def test_e2e_restart_resumes_schedule_into_new_attempt_dir(tmp_path):
    final, sup, run_dir, config = _supervise(tmp_path, fault="sigsegv@9", policy=dict(max_restarts=1))
    assert final["status"] == "COMPLETE_AFTER_RESTART" and final["classes"] == ["INFRA_CRASH_SIGNAL", "COMPLETE"]
    assert final["infra_failure_families"] == {"SEGFAULT_SIGNAL": 1}
    sup_records = S.read_ledger(run_dir / "supervisor.jsonl")[0]
    launches = [r for r in sup_records if r["type"] == "launch"]
    assert launches[0]["fault_inject"] == "sigsegv@9" and launches[1]["fault_inject"] is None
    assert launches[1]["resumed_from"] == ["attempt_001"]
    first, second = _attempt_records(run_dir, 1), _attempt_records(run_dir, 2)
    done1, next1 = S.completed_quota(first, 4)
    start2 = next(r for r in second if r["type"] == "start_segments")
    assert start2["completed_at_start"] == done1.tolist()
    resumed_first = {s["env"]: s["episode_index"] for s in start2["segments"] if s["kind"] == "episode"}
    for e, index in resumed_first.items():
        assert index == next1[e]                                  # schedule continues; incomplete index replayed
    total, _ = S.completed_quota(first + second, 4)
    assert (total >= 3).all()
    from analyze_long_rollout import analyze
    result = analyze(run_dir, config)
    assert result["infrastructure"]["process_failures"] == {"INFRA_CRASH_SIGNAL": 1}
    assert result["infrastructure"]["infra_incomplete_episodes"] == 4
    assert result["acceptance"]["checks"]["quota_every_env"] and result["acceptance"]["verdict"] == "PASS"


def test_e2e_termination_reason_info_is_read_through_adapter(tmp_path):
    final, _, run_dir, _ = _supervise(tmp_path, fault="nan_env:0@3",
                                      child_args=("--fake-info-style", "termination_reason"))
    assert final["status"] == "COMPLETE"
    records = _attempt_records(run_dir)
    adapter = next(r for r in records if r["type"] == "adapter")
    assert adapter["source"] == "termination_reason" and adapter["infra_key"] == "infra_error"
    bad = [r for r in records if r["type"] == "episode" and r["end_class"] == S.ENV_INVALID_STATE]
    assert len(bad) == 1 and bad[0]["adapter_source"] == "termination_reason"
    canary = [r for r in records if r["type"] == "episode" and r.get("canary")]
    assert any(r["canary"]["status"] == "FAIL" and r["env"] == 0 for r in canary)   # terminated canary = FAIL


def test_e2e_second_device_grid_marks_invalid_multi_grid_process(tmp_path):
    final, _, run_dir, config = _supervise(tmp_path, fault="second_grid@3")
    assert final["status"] == "COMPLETE"
    from analyze_long_rollout import analyze
    result = analyze(run_dir, config)
    b4 = result["attempts"][0]["bounds"]["B4"]
    assert b4["status"] == "FAIL" and b4["invalid_multi_grid_process"]
    assert result["acceptance"]["label"] == "INVALID_MULTI_GRID_PROCESS" and result["acceptance"]["verdict"] == "FAIL"


def test_fault_spec_parser_rejects_unknown_kinds():
    from run_long_rollout import parse_faults
    assert parse_faults("sigsegv@3, nan_env:2@7") == [dict(kind="sigsegv", step=3, env=None),
                                                      dict(kind="nan_env", step=7, env=2)]
    assert parse_faults(None) == []
    with pytest.raises(ValueError):
        parse_faults("explode@3")


def test_fault_injection_is_not_in_library_code():
    """The library never reads RCS_FAULT_INJECT or triggers faults; the supervisor only forwards the variable to
    attempt 1 of its child (tested above)."""
    lib = ROOT / "src" / "robotic_cutting_sim"
    hits = [p for p in lib.rglob("*.py") if any(t in p.read_text() for t in (
        'os.environ.get("RCS_FAULT_INJECT', 'os.environ["RCS_FAULT_INJECT"]', "def parse_faults", "FaultInjector"))]
    assert hits == []

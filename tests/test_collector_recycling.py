"""Opt-in process recycling of the dataset collector (``--recycle-every N``, WP-11 Phase H mitigation).

Kinematic CPU backend at coarse spacing: every block of N episodes runs in a fresh
child process; episode ids, seeds and the index stay continuous, every reset is
isolated, the dataset validates with ``read_dataset.py`` and the episodes are
bitwise identical (CPU) to the same plan collected in one process.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from robotic_cutting_sim import dataset as ds  # noqa: E402

PROGRAM = str(ROOT/"configs/wp11/action_programs/sever_mid_handwritten_v1.json")


def _plan():
    program = {"controller": "action_file", "action_file": PROGRAM, "seed": 3, "max_steps": 40}
    hold = {"controller": "policy", "policy": "robotic_cutting_sim.policy_examples:hold_policy", "seed": "random",
            "max_steps": 15}
    return {"schema": "rcs_episode_plan_v1", "episodes": [
        dict(program, label="program_a"), dict(hold, label="hold_a"), dict(program, label="program_b"),
        dict(hold, label="hold_b"), dict(program, label="program_c")]}


def _collect(tmp_path, name, extra=()):
    out = tmp_path/name
    command = [sys.executable, str(ROOT/"scripts/collect_dataset.py"), "--runtime", "kinematic", "--scene",
               "bar_160x30x20", "--goal", "sever_mid", "--spacing", "0.004", "--material-dt", "0.0005",
               "--device", "cpu", "--episode-plan", str(tmp_path/"plan.json"), "--record-every", "10",
               "--out", str(out), *extra]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=3000)
    return out, done


@pytest.mark.cpu_sim
def test_recycled_collection_is_continuous_isolated_and_matches_one_process(tmp_path):
    pytest.importorskip("warp")
    (tmp_path/"plan.json").write_text(json.dumps(_plan()))
    single, done_single = _collect(tmp_path, "single")
    recycled, done = _collect(tmp_path, "recycled", ["--recycle-every", "2"])
    # 0 = POLICY_READY, 3 = scored but not POLICY_READY (budgeted episodes); both mean every block ran
    assert done_single.returncode in (0, 3), done_single.stderr[-3000:]
    assert done.returncode in (0, 3), done.stderr[-3000:]

    index = ds.load_index(recycled)
    ids = [row["episode_id"] for row in index["episodes"]]
    assert ids == [f"episode_{i:03d}" for i in range(5)]
    assert [row["status"] for row in index["episodes"]] == ["STORED"]*5
    blocks = index["process_recycling"]["blocks"]
    assert [(b["block"], b["first_episode"], b["episodes"], b["returncode"]) for b in blocks] == \
        [(0, 0, 2, 0), (1, 2, 2, 0), (2, 4, 1, 0)]

    metas = [ds.load_episode(recycled, i)["meta"] for i in range(5)]
    singles = [ds.load_episode(single, i)["meta"] for i in range(5)]
    # each block in its own fresh process; never the parent's
    pids = [m["recycle"]["pid"] for m in metas]
    assert pids[0] == pids[1] and pids[2] == pids[3] and len({pids[0], pids[2], pids[4]}) == 3
    assert all(m["memory_after"]["pid"] == m["recycle"]["pid"] for m in metas)
    assert [m["recycle"]["block_episode"] for m in metas] == [0, 1, 0, 1, 0]
    assert all((recycled/f"runtime_block_{b:03d}").is_dir() for b in range(3))
    # continuous indexing and seeds: identical to the one-process collection of the same plan
    assert [m["reset_info"]["episode_index"] for m in metas] == list(range(5))
    assert [m["seed"] for m in metas] == [m["seed"] for m in singles]
    assert [m["seed_source"] for m in metas] == [m["seed_source"] for m in singles]
    assert metas[1]["seed"] != metas[3]["seed"]   # the random stream continues across processes
    manifest = json.loads((recycled/"dataset_manifest.json").read_text())
    assert manifest["config"]["process_recycling"] == {"every": 2, "blocks": 3}
    # state isolation: every reset clean, and the episodes are bitwise equal to the one-process run (CPU)
    for meta, other in zip(metas, singles):
        checks = meta["reset_checks"]
        assert checks["time_zero"] and checks["labels_clean"] and checks["control_steps_zero"]
        assert checks["material_codes_sha256"] == metas[0]["reset_checks"]["material_codes_sha256"]
        assert meta["final_codes_sha256"] == other["final_codes_sha256"]
        assert meta["task_id"] == other["task_id"] and meta["steps_recorded"] == other["steps_recorded"]
    assert len({metas[i]["final_codes_sha256"] for i in (0, 2, 4)}) == 1

    report = json.loads((recycled/"reliability.json").read_text())
    assert report["cross_episode"]["same_seed_same_controller_repeatable"] is True
    assert report["memory"]["trend_scope"] == "run"
    assert [b["episodes"] for b in report["memory"]["per_process"]["blocks"]] == \
        [["episode_000", "episode_001"], ["episode_002", "episode_003"], ["episode_004"]]
    assert "per_process" not in json.loads((single/"reliability.json").read_text())["memory"]
    assert "process_recycling" not in ds.load_index(single)
    reader = subprocess.run([sys.executable, str(ROOT/"scripts/read_dataset.py"), str(recycled), "--episode",
                             "episode_002", "--replay-kinematic"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert reader.returncode == 0 and '"replay": "PASS"' in reader.stdout, reader.stdout[-2000:]
    full = subprocess.run([sys.executable, str(ROOT/"scripts/read_dataset.py"), str(recycled)], cwd=ROOT,
                          capture_output=True, text=True, timeout=900)
    assert full.returncode == 0 and "READ_DATASET PASS" in full.stdout, full.stdout[-2000:]


def test_recycle_child_refuses_a_missing_dataset(tmp_path):
    spec = tmp_path/"block.json"
    spec.write_text("{}")
    done = subprocess.run([sys.executable, str(ROOT/"scripts/collect_dataset.py"), "--runtime", "kinematic",
                           "--scene", "bar_160x30x20", "--goal", "sever_mid", "--recycle-child", str(spec),
                           "--out", str(tmp_path/"missing")], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode != 0 and "needs the parent dataset directory" in done.stderr


def test_recycle_blocks_split_without_gaps():
    sys.path.insert(0, str(ROOT/"scripts"))
    import collect_dataset as cd
    blocks = cd.recycle_blocks(list(range(7)), 3)
    assert [(b, first, list(e)) for b, first, e in blocks] == [(0, 0, [0, 1, 2]), (1, 3, [3, 4, 5]), (2, 6, [6])]


def _meta(i, block, rss, gpu=None):
    reset = {"time_zero": True, "labels_clean": True, "material_at_reference_max_m": 0., "material_velocity_max_m_s": 0.,
             "material_codes_sha256": "z", "oracle_history_single_sample": True, "lifecycle_ready_after_reset": True,
             "ik_failures_zero": True, "not_terminated": True, "control_steps_zero": True, "action_log_empty": True,
             "observation_timestamp_matches": True, "tool_pose_world_xyzw": [0, 0, 0, 0, 0, 0, 1],
             "tool_linear_speed_m_s": 0., "tool_angular_speed_rad_s": 0., "grasp_reset_status": None,
             "camera_timestamp_matches": None}
    return {"episode_id": f"episode_{i:03d}", "label": str(i), "controller": "scripted", "seed": 42,
            "controller_receipt": {"kind": "scripted"}, "verdict": "FAIL" if i == 0 else "PASS", "expect": None,
            "reset_checks": reset, "clock_errors": 0, "observation_privileged_leaks": 0, "recorder_closed": True,
            "final_codes_sha256": "a" if i else "b", "runtime_options": {"device": "cpu"},
            "memory_after": {"pid": 100+block, "host_rss_mib": rss, "gpu_used_mib_total": gpu},
            "preset": {"camera_every": 1}, "control_dt_s": .02, "camera_age_max_s": 0.,
            "recycle": {"block": block, "every": 3}}


def test_per_process_trend_scope_is_opt_in_and_fits_within_each_process():
    v1a = json.loads((ROOT/"configs/wp11/phaseG_reliability_v1a.json").read_text())
    # each process grows 10 MiB/episode but restarts low: the whole-run fit sees a sawtooth
    rss = [1000., 1010., 1020., 1000., 1010., 1020.]
    metas = [_meta(i, i//3, rss[i]) for i in range(6)]
    metas[0]["controller"] = "policy"   # a failed episode of another controller (stored and usable)
    validations = [{"episode_id": m["episode_id"], "status": "PASS"} for m in metas]
    run = ds.reliability_report(metas, v1a, "kinematic", validations=validations)
    assert run["memory"]["trend_scope"] == "run"
    assert run["memory"]["host_rss_slope_mib_per_episode"] == pytest.approx(ds._slope(rss[1:]))
    blocks = run["memory"]["per_process"]["blocks"]
    assert [b["host_rss_slope_mib_per_episode"] for b in blocks] == [pytest.approx(10.), pytest.approx(10.)]
    assert run["memory"]["per_process"]["host_rss_max_mib"] == 1020.
    scoped = dict(v1a, memory=dict(v1a["memory"], trend_scope="per_process", host_rss_slope_max_mib_per_episode=5))
    report = ds.reliability_report(metas, scoped, "kinematic", validations=validations)
    assert report["memory"]["host_rss_slope_mib_per_episode"] == pytest.approx(10.)
    assert "host_memory_no_leak" in report["failing"]
    plain = [{k: v for k, v in m.items() if k != "recycle"} for m in metas]
    assert "per_process" not in ds.reliability_report(plain, v1a, "kinematic", validations=validations)["memory"]
    with pytest.raises(ValueError):
        ds.reliability_report(plain, scoped, "kinematic", validations=validations)


_HARD_EXIT_CHILD = r'''
import contextlib, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import collect_dataset as cd

@contextlib.contextmanager
def fake_env(*args, **kwargs):
    yield object()
    os._exit(0)   # what Kit's SimulationApp.close does on Native: the process ends inside the env close

def fake_episode(env, entry, index, out, a, clocks, extra=None):
    meta = dict(episode_id=f"episode_{index:03d}", label=entry["label"], controller=entry["controller"],
                seed=entry["seed"], verdict="PASS", expect=None, termination_reason="TIME_LIMIT", home_outcome=None,
                steps_recorded=1, export=dict(sha256="x"), wall_s=0., action_status_counts={},
                memory_after=dict(host_rss_mib=1.), **(extra or {}))
    return meta

cd.open_env, cd.run_episode = fake_env, fake_episode
sys.exit(cd.main(sys.argv[2:]))
'''


def test_recycle_child_reports_its_episodes_even_when_env_close_ends_the_process(tmp_path):
    """Regression (WP-11 Phase H mitigation run 001): on Native the env close (Kit SimulationApp.close) ends the
    child process, so the block result must already be on disk when the env context exits."""
    plan = {"schema": "rcs_episode_plan_v1", "episodes": [
        {"label": f"hold_{i}", "controller": "policy", "policy": "robotic_cutting_sim.policy_examples:hold_policy",
         "seed": 7} for i in range(2)]}
    (tmp_path/"plan.json").write_text(json.dumps(plan))
    out = tmp_path/"dataset"
    (out/"recycle").mkdir(parents=True)
    (out/"dataset_manifest.json").write_text("{}")
    entries = [dict(e, seed_source="fixed", edge_feedback="nominal") for e in plan["episodes"]]
    spec = out/"recycle"/"block_000.json"
    spec.write_text(json.dumps(dict(block=0, every=2, first_episode=0, episodes=2, compile_seed=7,
                                    runtime_dir="runtime_block_000", entries=entries)))
    (tmp_path/"child.py").write_text(_HARD_EXIT_CHILD)
    done = subprocess.run([sys.executable, str(tmp_path/"child.py"), str(ROOT/"scripts"), "--runtime", "kinematic",
                           "--scene", "bar_160x30x20", "--goal", "sever_mid", "--episode-plan", str(tmp_path/"plan.json"),
                           "--recycle-every", "2", "--recycle-child", str(spec), "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads((out/"recycle"/"block_000_result.json").read_text())
    assert [row["episode_id"] for row in result["index_rows"]] == ["episode_000", "episode_001"]
    assert [row["status"] for row in result["index_rows"]] == ["STORED", "STORED"]
    assert result["env_closed"] is False and result["status"] == 0 and result["failures"] == []

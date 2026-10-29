"""CPU tests of scripts/summarize_phase_f.py on synthetic run directories (no simulation).

The summarizer must copy verdicts unchanged, label ERROR / supplementary rows, extract
the traceback summary of a raised row from its log, cross-check queue verdicts and
protocol digests, and never overwrite an existing evidence file.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_phase_f.py"
PROTOCOL_SHA = hashlib.sha256((ROOT / "configs/wp11/phaseF_protocol_v1.json").read_bytes()).hexdigest()


def _module():
    spec = importlib.util.spec_from_file_location("summarize_phase_f", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(trim=False):
    m = {"final_mechanical_components": 2,
         "components": {"component_mass_fraction": [.5, .5], "component_mass_kg": [.072, .072]},
         "mass_check": {"recorded_total_mass_kg": .144, "pass": True},
         "final_labelled_count": 10, "max_labelled_count": 10, "labelled_count_history": list(range(20)),
         "component_count_history": [1, 1, 2, 2],
         "label_to_swept_surface": {"0": {"label_distance": {"labelled_particles": 10}, "interface": {
             "maximum_reference_midpoint_plane_distance_m": .0015}, "ideal_partition_interface_floor_m": .0015}},
         "swept_planes": {"0": {"status": "FIT", "fit": {"normal": [1., 0., 0.], "rms_residual_m": 0.}}},
         "forces": {"cut": {"steps": 5, "total_force_mean_N": .01, "total_force_max_N": .05, "steps_with_blade_contacts": 5}},
         "blade_reaction_episode": {"mean_N_contact_steps": .02},
         "plowing": {"plowing_step_fraction_of_cutting_steps": 0.}, "stroke": {},
         "stability": {"minimum_det_F": .94, "inverted_steps": 0, "stable": True, "scope": "x"},
         "bridges": [], "workspace_clamped_steps": 0}
    if trim:
        m["trim_separation"] = {"status": "PASS_TRIM_SEPARATION", "metrics": {
            "removed_reference_above_plane_mass_fraction": .95, "largest_component_below_plane_mass_share": 1.,
            "detached_below_plane_mass_fraction": 0.}}
    else:
        m["frozen_strict"] = {"status": "PASS_TASK_METRICS", "checks": {"cut_at_workpiece_local_target_plane": True},
                              "metrics": {"maximum_reference_midpoint_plane_distance_m": .0015}}
    return m


def _run(base, name, row, variant, verdict, *, protocol_sha=PROTOCOL_SHA, error=None, trim=False, dt=1e-4):
    path = base / "runs" / name
    path.mkdir(parents=True)
    result = {"row": row, "variant": variant, "protocol_row": "F3_resolution" if row.startswith("F3") else row,
              "verdict": verdict, "criterion_outcome": None if error else "PASS", "checks": None if error else {"x": True},
              "git_sha": "abc", "git_dirty": False, "protocol_sha256": protocol_sha, "seed": 42, "device": "cuda:0",
              "particle_spacing_m": .001, "grid_dx_m": .004, "material_dt_s": dt, "wall_s": 12.3456, "context": {}}
    if error:
        result["error"] = error
    else:
        result["metrics"] = _metrics(trim)
    (path / "result.json").write_text(json.dumps(result))
    (path / "manifest.json").write_text("{}")
    return path


ERROR_LOG = """Traceback (most recent call last):
  File "/w/scripts/run_phase_f_material.py", line 849, in <module>
    main()
  File "/w/src/robotic_cutting_sim/coupled_robot.py", line 37, in __post_init__
    raise ValueError("control_dt must be an integer multiple of material_dt")
ValueError: control_dt must be an integer multiple of material_dt
Command exited with non-zero status 1
\tCommand being timed: "python scripts/run_phase_f_material.py --row F3_cut --variant medium"
\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:00.26
\tExit status: 1
"""


@pytest.fixture()
def tree(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    runs = [_run(tmp_path, "f3_cut_formal_x", "F3_cut", "formal", "PASS"),
            _run(tmp_path, "f3_cut_medium_x", "F3_cut", "medium", "ERROR", error="ValueError('control_dt ...')"),
            _run(tmp_path, "f3_cut_medium_dt133_supplementary_x", "F3_cut", "medium_dt133_supplementary",
                 "CHARACTERIZATION", dt=.02/133),
            _run(tmp_path, "f2_1_gate_14_x", "F2.1_gate_14", "formal", "PASS", trim=True, protocol_sha="0"*64)]
    (logs / "phaseF_f3_cut_medium_x.log").write_text(ERROR_LOG)
    (logs / "phaseF_f3_cut_formal_x.log").write_text("\tExit status: 0\n")
    queue = tmp_path / "queue.log"
    queue.write_text("START t F3_cut formal f3_cut_formal_x\n"
                     "END t F3_cut formal f3_cut_formal_x rc=0 F3_cut formal FAIL FAIL {'x': True} 12s\n"
                     "END t F3_cut medium f3_cut_medium_x rc=0 \n")
    return tmp_path, runs, logs, queue


def test_summarize_rows_copy_verdicts_and_label_error_and_supplementary(tree):
    base, runs, logs, queue = tree
    S = _module()
    ends = S._queue_ends(queue)
    rows = {r["run"]: r for r in (S.summarize(p, logs, ends, PROTOCOL_SHA) for p in runs)}
    formal, error, supp, gate = (rows[n] for n in ("f3_cut_formal_x", "f3_cut_medium_x",
                                                   "f3_cut_medium_dt133_supplementary_x", "f2_1_gate_14_x"))
    assert formal["verdict"] == "PASS" and formal["protocol_status"] == "PROTOCOL_PASS_FAIL"
    assert formal["queue_verdict_matches_result"] is False          # queue text says FAIL: flagged, not rescored
    assert error["verdict"] == "ERROR" and error["protocol_status"] == "ERROR_NOT_RUN"
    assert error["error"].startswith("ValueError")
    assert error["log"]["exit_status"] == 1 and error["queue_end"]["rc"] == 0
    tb = error["log"]["traceback_summary"]
    assert tb["exception"] == "ValueError: control_dt must be an integer multiple of material_dt"
    assert tb["innermost_frames"][-1].endswith("in __post_init__")
    assert supp["protocol_status"] == "SUPPLEMENTARY_NON_PROTOCOL" and supp["verdict"] == "CHARACTERIZATION"
    assert gate["protocol_sha256_matches_protocol_file"] is False
    assert formal["protocol_sha256_matches_protocol_file"] is True
    assert "manifest.json" in formal["inputs_sha256"] and "log" in error["inputs_sha256"]
    table = S.f3_table(sorted(rows.values(), key=S._order))
    assert [t["variant"] for t in table] == ["formal", "medium", "medium_dt133_supplementary"]
    assert table[0]["frozen_strict"] == "PASS_TASK_METRICS" and table[0]["cut_location_midpoint_statistic_m"] == .0015
    assert table[1]["error"].startswith("ValueError")


def test_cli_writes_once_and_refuses_overwrite(tree, tmp_path):
    base, runs, logs, queue = tree
    out = tmp_path / "evidence.json"
    command = [sys.executable, str(SCRIPT), "--runs", *map(str, runs), "--logs", str(logs),
               "--queue-log", str(queue), "--out", str(out)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    evidence = json.loads(out.read_text())
    assert [r["run"] for r in evidence["rows"]][0] == "f2_1_gate_14_x"   # protocol order, not lexical
    integrity = evidence["integrity"]
    assert integrity["all_protocol_sha256_match"] is False
    assert integrity["queue_rc0_but_log_exit_nonzero"] == ["f3_cut_medium_x"]
    assert integrity["queue_verdict_mismatches"] == ["f3_cut_formal_x"]
    supp = next(t for t in evidence["f3_resolution_table"] if t["variant"] == "medium_dt133_supplementary")
    assert supp["material_dt_s"] == pytest.approx(.02/133, rel=1e-12)   # not rounded to the declared 0.15 ms
    before = out.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0 and "refusing to overwrite" in second.stderr
    assert out.read_bytes() == before

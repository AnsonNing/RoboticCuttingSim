"""Matrix receipt semantics with synthetic files; no simulator success claim."""
import json
from pathlib import Path
import runpy

from robotic_cutting_sim.recording import ROOT

summarize = runpy.run_path(str(ROOT/"scripts/run_cutting_task_benchmark.py"))["summarize_case"]


def test_completed_episode_can_pass_separation_and_fail_surface_goal(tmp_path):
    (tmp_path/"result.json").write_text(json.dumps({"status": "PASS_EXECUTION"}))
    checks = dict.fromkeys(["complete_episode", "initial_continuous_graph", "initial_codes_zero",
        "reference_matches_single_continuous_task_sampler", "reference_mass_budget_and_unchanged_count",
        "persistent_history_no_digit_rewrites", "postcut_disconnected_topology_persists",
        "reference_above_plane_material_removed"], True)
    checks["surface_p95_absolute_plane_error_m"] = False
    (tmp_path/"task_evaluation.json").write_text(json.dumps({"status": "FAIL_TASK_METRICS", "checks": checks}))
    row = summarize(tmp_path, "TRIM-01", "isaac_lab", 0)
    assert row["episode_execution"] == "PASS" and row["physical_cutting"] == "PASS"
    assert row["task_goal_v1"] == "FAIL_TASK_METRICS"
    assert row["camera_acceptance"] == "SEPARATE_AUDIT_REQUIRED" and not row["full_simulation_acceptance"]
    # A process failure after writing nominal success must not pass the matrix.
    failed = summarize(tmp_path, "TRIM-01", "isaac_lab", 1)
    assert failed["episode_execution"] == "FAIL" and failed["physical_cutting"] == "NOT_RUN"


def test_partial_runtime_smoke_never_scores_final_physical_success(tmp_path):
    (tmp_path/"result.json").write_text(json.dumps({"status": "PASS_SMOKE"}))
    row = summarize(tmp_path, "CUT-01", "isaac_sim", 0)
    assert row["physical_cutting"] == "NOT_RUN" and row["task_goal_v1"] == "NOT_RUN"

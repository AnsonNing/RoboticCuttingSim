"""Portable preflight, physical evidence and expected-failure boundaries."""
from copy import deepcopy
from pathlib import Path
import venv

from robotic_cutting_sim.deployment_doctor import (
    REQUIRED, assess_runtime, classify_pip_check, parse_gpu_csv, probe_runtime, select_gpu,
)
from robotic_cutting_sim.deployment_validation import CAMERA_IDS, classify_episode, make_plan, summarize_validation


def test_physical_deployment_uses_both_real_runtimes_and_both_tasks_without_promoting_smoke():
    plan = make_plan("/tmp/repo", "/tmp/output", tool_attachment_mode="physical_grasp",
        initial_finger_position_m=.0112, gpu=0)
    episodes = [s for s in plan if s["kind"] == "episode"]
    assert len(episodes) == 4
    assert {(s["runtime"],s["task_id"]) for s in episodes} == {
        (r,t) for r in ("isaac_sim","isaac_lab") for t in ("CUT-01","TRIM-01")}
    for stage in episodes:
        assert stage["steps"] == 2 and stage["task_audit_command"] is None
        assert "physical_grasp" in stage["command"] and "0.0112" in stage["command"]
        assert stage["physical_tool_audit_command"]
    legacy = make_plan("/tmp/repo", "/tmp/output")
    assert len([s for s in legacy if s["kind"] == "episode"]) == 2
    assert all(s["physical_tool_audit_command"] is None for s in legacy if s["kind"] == "episode")


def test_gpu_auto_prefers_free_rtx_and_supports_a_single_physical_gpu():
    rows = parse_gpu_csv("0, GPU-a, NVIDIA RTX 5090, 32607, 29000, 570.211.01, 12.0, 0\n"
                         "1, GPU-b, NVIDIA RTX 5090, 32607, 18000, 570.211.01, 12.0, 5\n"
                         "2, GPU-c, NVIDIA A100, 80000, 79000, 570.211.01, 8.0, 0")
    assert select_gpu(rows)["physical_index"] == 0
    assert select_gpu(rows[:1])["cuda_device"] == "cuda:0"
    assert select_gpu(rows, "1")["physical_index"] == 1
    assert select_gpu(rows[:1], "1")["status"] == "BLOCKED_REQUESTED_GPU_ABSENT"
    assert select_gpu([])["status"] == "BLOCKED_NO_NVIDIA_GPU"
    assert not select_gpu(rows)["native_compatibility_validated"]


def test_pip_optional_warnings_preserve_nonzero_and_unexpected_lines():
    pip = {"returncode": 1, "stdout": "mujoco-usd-converter 0.6.1 requires usd-exchange, which is not installed.\nosqp 1.0.5 requires joblib, which is not installed.", "stderr": ""}
    result = classify_pip_check("isaac_sim", pip)
    assert result["status"] == "WARN_EXPECTED_OPTIONAL_METADATA" and result["returncode"] == 1
    bad = {**pip, "stdout": pip["stdout"]+"\nnumpy 2.3 requires unknown, which is not installed."}
    assert classify_pip_check("isaac_sim", bad)["status"] == "WARN_UNEXPECTED_METADATA"


def metadata(runtime):
    return {"python_version_info": [3, 12, 12], "packages": {k: v or "1.0" for k, v in REQUIRED[runtime].items()}}


def test_usd_provider_pollution_and_required_versions_are_blocked():
    good = metadata("isaac_sim")
    assert assess_runtime("isaac_sim", good, {})["status"] == "PASS_METADATA_PRECHECK"
    polluted = deepcopy(good);polluted["packages"]["usd-exchange"] = "3.0.1"
    assert assess_runtime("isaac_sim", polluted, {})["usd_provider_conflict"]
    wrong = deepcopy(good);wrong["packages"]["torch"] = "2.11.0+cu130"
    assert assess_runtime("isaac_sim", wrong, {})["status"].startswith("BLOCKED")
    wrong = deepcopy(good);wrong["python_version_info"] = [3, 10, 12]
    assert not assess_runtime("isaac_sim", wrong, {})["python_3_12"]
    assert "imageio" not in REQUIRED["base"]


def test_foreign_or_missing_project_origin_is_blocked(tmp_path):
    good = metadata("base")
    good["project_import_origin"] = str(tmp_path/"src/robotic_cutting_sim/__init__.py")
    assert assess_runtime("base", good, {}, project_root=tmp_path)["project_origin_matches_checkout"]
    good["project_import_origin"] = None
    assert assess_runtime("base", good, {}, project_root=tmp_path)["status"].startswith("BLOCKED")
    good["project_import_origin"] = str(tmp_path/"another_checkout/src/robotic_cutting_sim/__init__.py")
    assert assess_runtime("base", good, {}, project_root=tmp_path)["status"].startswith("BLOCKED")


def test_probe_ignores_injected_python_paths_without_project_install(tmp_path, monkeypatch):
    # A genuine empty venv must not inherit an apparently valid project origin
    # from PYTHONPATH, nor fail to start because of a foreign PYTHONHOME.
    environment = tmp_path/"empty-venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    root = Path(__file__).resolve().parents[1]
    foreign = tmp_path/"foreign"
    (foreign/"robotic_cutting_sim").mkdir(parents=True)
    (foreign/"robotic_cutting_sim/__init__.py").write_text("raise AssertionError('not executed')\n")
    monkeypatch.setenv("PYTHONHOME", str(tmp_path/"nonexistent-python-home"))
    for injected in (root/"src", foreign):
        monkeypatch.setenv("PYTHONPATH", str(injected))
        result = probe_runtime("base", environment/"bin/python", root=root)
        assert result["status"].startswith("BLOCKED")
        assert result["metadata"]["project_import_origin"] is None
        assert result["project_origin_matches_checkout"] is False
        assert result["pip_check"]["raw"]["command"][1:4] == ["-I", "-m", "pip"]


def test_plan_has_no_second_gpu_dependency_and_retains_trim_override(tmp_path):
    plan = make_plan(tmp_path/"root with spaces", tmp_path/"outputs", tier="full", gpu=0)
    episodes = [p for p in plan if p["kind"] == "episode"]
    assert len(episodes) == 6
    for case in episodes:
        command = case["command"]
        assert command[command.index("--device")+1] == "cuda:0"
        assert "cuda:1" not in command
        if case["runtime"] == "isaac_sim":assert command[command.index("--gpu")+1] == "0"
        if case["task_id"] == "TRIM-01":assert command[command.index("--particle-spacing")+1] == "0.002"


def valid_receipts(steps=400, task_id="TRIM-01"):
    result = {"status": "PASS_EXECUTION" if steps == 400 else "PASS_SMOKE", "steps": steps,
        "task_id": task_id,
        "simulation_s": .02*steps, "robot_dynamics_source": "Isaac PhysX", "native_physics_steps": 10*steps,
        "standalone_robot_dynamics_steps": 0, "replay_inputs": False, "camera_mode": "native_single_view",
        "reset_receipt": {"dynamics_reset_measured": True, "q_max_error_rad": 0., "all_nine_dofs_max_error_rad": 0.,
             "all_body_pose_max_abs_error": 0., "material_max_error_m": 0., "nonzero_cut_codes": 0, "time_s": 0.}}
    camera_checks = {name+suffix: True for name in CAMERA_IDS for suffix in
                     ("_finite_range_and_masks", "_calibration_measured_pose", "_nonempty_working_depth", "_runtime_optics")}
    camera_checks["frame_timestamps"] = True
    count = 41 if steps == 400 else 3
    camera = {"kinematics": {"status": "PASS"}, "all_manifest_implementation_bytes_resolved": True,
        "frames": count, "records": [{"cameras": dict.fromkeys(CAMERA_IDS, {})} for _ in range(count)],
        "visibility_and_optics": {"status": "FAIL", "checks": camera_checks}, "complete_8s_episode": steps == 400}
    score_checks = dict.fromkeys(("complete_episode", "initial_continuous_graph", "initial_codes_zero",
        "reference_matches_single_continuous_task_sampler", "reference_mass_budget_and_unchanged_count",
        "persistent_history_no_digit_rewrites", "postcut_disconnected_topology_persists",
        "reference_above_plane_material_removed"), True)
    score_checks["surface_p95_absolute_plane_error_m"] = task_id != "TRIM-01"
    task = {"task_score": {"status": "FAIL_TASK_METRICS" if task_id == "TRIM-01" else "PASS_TASK_METRICS",
        "checks": score_checks, "metrics": {"initial_components": 1, "final_components": 2,
            "nonzero_digit_rewrites": 0, "reference_mass_absolute_error_kg": 0., "postcut_persistence_time_s": 2.4}},
        "all_manifest_implementation_bytes_resolved": True,
        "recorded_score_recomputation": {"same_checks": True, "same_metrics": True, "same_status": True},
        "contact_and_numerical_screens": {"checks": {"finite": True}}}
    return result, camera, task


def classify(receipts, steps=400, task_id="TRIM-01", returncode=0):
    return classify_episode(*receipts, runtime="isaac_sim", expected_steps=steps, returncode=returncode, task_id=task_id,
                            reset=(receipts[0] or {}).get("reset_receipt"))


def test_reset_or_wrong_task_cannot_be_masked_by_process_success():
    result, camera, task = valid_receipts(2)
    result["reset_receipt"]["nonzero_cut_codes"] = 1
    assert classify((result, camera, task), steps=2)["status"] == "FAIL_RUNTIME_CONTRACT"
    result["reset_receipt"]["nonzero_cut_codes"] = 0;result["task_id"] = "WRONG-TASK"
    assert classify((result, camera, task), steps=2)["status"] == "FAIL_RUNTIME_CONTRACT"
    result["task_id"] = "TRIM-01";result.pop("reset_receipt")
    assert classify((result, camera, task), steps=2)["status"] == "FAIL_RUNTIME_CONTRACT"


def test_exit_zero_missing_receipts_does_not_validate_runtime():
    assert classify((None, None, None))["status"] == "FAIL_RUNTIME_CONTRACT"
    result, camera, task = valid_receipts()
    result["native_physics_steps"] = 0
    assert classify((result, camera, task))["status"] == "FAIL_RUNTIME_CONTRACT"
    result["simulation_s"] = None
    assert classify((result, camera, task))["status"] == "FAIL_RUNTIME_CONTRACT"


def test_smoke_cannot_be_promoted_to_physical_success():
    outcome = classify(valid_receipts(2), steps=2)
    assert outcome["status"] == "PASS_REQUESTED_RUNTIME_CONTRACT"
    assert outcome["physical_cutting"] == "NOT_RUN_SMOKE"
    result, camera, task = valid_receipts(2)
    camera["records"][0]["cameras"].pop("global_d455")
    assert classify((result, camera, task), steps=2)["status"] == "FAIL_RUNTIME_CONTRACT"


def test_replay_or_ideal_native_substitution_fails_contract():
    result, camera, task = valid_receipts()
    result["replay_inputs"] = True
    assert classify((result, camera, task))["status"] == "FAIL_RUNTIME_CONTRACT"
    result["replay_inputs"] = False;result["camera_mode"] = "rtx_rgbd"
    assert classify((result, camera, task))["status"] == "FAIL_RUNTIME_CONTRACT"


def test_only_expected_trim_geometry_failure_can_be_allowed():
    outcome = classify(valid_receipts())
    assert outcome["physical_cutting"] == "PASS_SAMPLED_SEPARATION" and outcome["expected_trim_surface_failure"]
    stages = [{"kind": "static", "status": "PASS"}, {"kind": "episode", "status": "PASS", "classification": outcome}]
    assert not summarize_validation(stages, tier="full")["requested_scope_pass"]
    allowed = summarize_validation(stages, tier="full", allow_expected_trim_geometry_fail=True)
    assert allowed["requested_scope_pass"] and not allowed["all_strict_full_task_metrics_pass"]
    assert not allowed["full_benchmark_ready"]
    result, camera, task = valid_receipts()
    task["task_score"]["checks"]["reference_above_plane_material_removed"] = False
    failed = classify((result, camera, task))
    assert not failed["expected_trim_surface_failure"] and failed["physical_cutting"] == "FAIL"


def test_mass_or_history_failure_cannot_use_trim_exception():
    result, camera, task = valid_receipts()
    task["task_score"]["metrics"]["nonzero_digit_rewrites"] = 1
    assert not classify((result, camera, task))["expected_trim_surface_failure"]
    task["task_score"]["metrics"]["nonzero_digit_rewrites"] = 0
    task["task_score"]["metrics"]["reference_mass_absolute_error_kg"] = .0001
    assert not classify((result, camera, task))["expected_trim_surface_failure"]


def test_unexecuted_static_or_full_scope_cannot_pass():
    assert not summarize_validation([], tier="static")["requested_scope_pass"]
    assert not summarize_validation([{"kind": "cpu", "status": "SKIPPED_BY_REQUEST"}], tier="static")["requested_scope_pass"]
    smoke = classify(valid_receipts(2), steps=2)
    stages = [{"kind": "static", "status": "PASS"}, {"kind": "episode", "status": "PASS", "classification": smoke}]
    assert not summarize_validation(stages, tier="full")["requested_scope_pass"]

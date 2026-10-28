"""Pure portable-validation plans and evidence classifiers; no simulator imports."""
from pathlib import Path

from .deployment_doctor import RUNTIME_PATHS


CAMERA_IDS = {"wrist_d405_0", "wrist_d405_1", "global_d455"}
SURFACE_CHECKS = {"maximum_substrate_surface_residual_height_m", "surface_p95_absolute_plane_error_m",
                  "surface_rms_plane_error_m", "maximum_gouge_depth_m", "overcut_volume_fraction_of_initial_base",
                  "missing_surface_fraction"}


def _number(value, default=-1.):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def make_plan(root, output, *, tier="smoke", runtime="both", gpu=0, pythons=None, seed=42,
              tool_attachment_mode="legacy_fixed_tool", initial_finger_position_m=None,
              lab_grip_contact_profile="legacy_point_contact"):
    if tool_attachment_mode not in ("legacy_fixed_tool", "physical_grasp"):
        raise ValueError("Unsupported tool attachment mode")
    root, output = Path(root).resolve(), Path(output).resolve()
    pythons = {k: str(Path(v).expanduser().absolute()) for k, v in (pythons or {}).items()}
    python = lambda name: pythons.get(name, str(root/RUNTIME_PATHS[name]))
    base = python("base")
    stages = [{"name": "cpu_regression", "kind": "cpu", "runtime": "base", "output_dir": None,
               "command": [base, "-m", "pytest", "-q"]},
              {"name": "static_workcell", "kind": "static", "runtime": "base", "output_dir": str(output/"static_workcell"),
               "command": [base, "-m", "robotic_cutting_sim.cli", "scene", "--preset", "workcell",
                   "--camera-layout", "wrist_d405_global_d455", "--seed", str(seed), "--output-dir", str(output/"static_workcell")]}]
    selected = ["isaac_sim", "isaac_lab"] if runtime == "both" else [runtime]
    if tier == "static":
        return stages
    for backend in selected:
        cases = [("smoke_cut", "CUT-01", 2)]
        if tool_attachment_mode == "physical_grasp":
            cases += [("smoke_trim", "TRIM-01", 2)]
        if tier == "full":
            cases += [("full_cut", "CUT-01", 400), ("full_trim", "TRIM-01", 400)]
        for suffix, task, steps in cases:
            name = backend+"_"+suffix
            directory = output/name
            runner = root/"scripts"/("run_isaac_sim_cutting.py" if backend == "isaac_sim" else "run_isaac_lab_cutting.py")
            command = [python(backend), str(runner), "--task-id", task, "--steps", str(steps), "--seed", str(seed),
                "--camera-layout", "wrist_d405_global_d455", "--device", f"cuda:{gpu}", "--record-every", "1" if steps == 2 else "10",
                "--output-dir", str(directory)]
            if backend == "isaac_sim":
                command += ["--gpu", str(gpu), "--camera-mode", "native_single_view", "--native-noise", "off"]
            else:
                command += ["--material-device", f"cuda:{gpu}", "--reset-check"]
                command += ["--grip-contact-profile",lab_grip_contact_profile]
            if task == "TRIM-01":
                command += ["--particle-spacing", "0.002"]
            command += ["--tool-attachment-mode", tool_attachment_mode]
            if initial_finger_position_m is not None:
                command += ["--grip-initial-position", str(initial_finger_position_m)]
            stages.append({"name": name, "kind": "episode", "runtime": backend, "task_id": task, "steps": steps,
                "output_dir": str(directory), "command": command,
                "camera_audit_dir": str(output/(name+"_camera_audit")),
                "camera_audit_command": [base, str(root/"scripts/audit_wrist_camera_rig.py"), "--episode", str(directory),
                                         "--output-dir", str(output/(name+"_camera_audit"))],
                "task_audit_dir": str(output/(name+"_task_audit")),
                "task_audit_command": [base, str(root/"scripts/audit_cutting_task_episodes.py"), "--recording-dir", str(directory),
                                       "--output-dir", str(output/(name+"_task_audit"))] if steps == 400 else None,
                "physical_tool_audit_dir": str(output/(name+"_physical_audit")),
                "physical_tool_audit_command": [base,str(root/"scripts/audit_physical_tool.py"),str(directory),
                    str(output/(name+"_physical_audit"))] if tool_attachment_mode == "physical_grasp" else None})
    return stages


def classify_episode(result, camera, task, *, runtime, expected_steps, returncode, task_id, reset=None):
    """Exit zero alone cannot satisfy any live runtime or physical scope."""
    result, camera = result or {}, camera or {}
    checks = {"process_exit_zero": returncode == 0,
        "actual_result_receipt": result.get("status") in ("PASS_SMOKE", "PASS_EXECUTION"),
        "requested_task_id": result.get("task_id") == task_id,
        "expected_control_steps": result.get("steps") == expected_steps,
        "positive_expected_simulation_time": abs(_number(result.get("simulation_s"))-expected_steps*.02) <= 1e-6,
        "camera_audit_receipt": bool(camera), "camera_fk_and_timestamps": camera.get("kinematics", {}).get("status") == "PASS",
        "camera_source_bytes_resolved": camera.get("all_manifest_implementation_bytes_resolved") is True,
        "enough_actual_camera_frames": camera.get("frames", 0) >= (41 if expected_steps == 400 else 3)}
    if runtime == "isaac_sim":
        reset = reset or {}
        checks.update(actual_native_articulation=result.get("robot_dynamics_source") == "Isaac PhysX",
            actual_native_substeps=_number(result.get("native_physics_steps")) >= expected_steps*10,
            no_standalone_robot_dynamics=result.get("standalone_robot_dynamics_steps") == 0,
            no_replay=result.get("replay_inputs") is False,
            actual_native_pipeline=result.get("camera_mode") == "native_single_view",
            measured_reset_receipt=reset.get("dynamics_reset_measured") is True,
            reset_joints_restored=0. <= _number(reset.get("q_max_error_rad")) <= 1e-6,
            reset_all_dofs_restored=0. <= _number(reset.get("all_nine_dofs_max_error_rad")) <= 1e-6,
            reset_bodies_restored=0. <= _number(reset.get("all_body_pose_max_abs_error")) <= 1e-6,
            reset_material_restored=0. <= _number(reset.get("material_max_error_m")) <= 1e-6,
            reset_cut_history_cleared=reset.get("nonzero_cut_codes") == 0,
            reset_clock_zero=abs(_number(reset.get("time_s"))) <= 1e-9)
    else:
        reset = reset or result.get("reset_check") or {}
        checks.update(actual_lab_scene_env_task=result.get("actual_scene_environment_task_api") is True,
            actual_lab_source=result.get("source") == "live_isaac_lab",
            actual_lab_runtime_clock=abs(_number(result.get("runtime_s"))-expected_steps*.02) <= 1e-6,
            measured_reset_receipt=bool(reset),
            reset_joints_restored=0. <= _number(reset.get("joint_max_abs_error_rad")) <= 1e-6,
            reset_material_restored=0. <= _number(reset.get("material_max_abs_error_m")) <= 1e-6,
            reset_cut_history_cleared=reset.get("nonzero_cut_codes") == 0)
    records = camera.get("records", [])
    views = camera.get("visibility_and_optics", {}).get("checks", {})
    checks["named_rgbd_streams"] = bool(records) and all(set(row.get("cameras", {})) == CAMERA_IDS for row in records)
    checks["camera_count_matches_payloads"] = len(records) == camera.get("frames")
    optional_visibility = set(camera.get("optional_visibility_camera_ids", [])) & {"wrist_d405_1"}
    checks["camera_payload_range_calibration_contract"] = bool(records) and all(
        views.get(name+suffix) is True for name in CAMERA_IDS
        for suffix in ("_finite_range_and_masks", "_calibration_measured_pose", "_nonempty_working_depth")
        if suffix != "_nonempty_working_depth" or name not in optional_visibility)
    checks["rgbd_frame_timestamps"] = views.get("frame_timestamps") is True
    checks["wrist_runtime_optics"] = all(views.get(n+"_runtime_optics") is True for n in CAMERA_IDS if n.startswith("wrist"))
    if expected_steps == 400:
        checks["complete_camera_episode"] = camera.get("complete_8s_episode") is True
    score = (task or {}).get("task_score", {})
    metrics, score_checks = score.get("metrics", {}), score.get("checks", {})
    physical = (expected_steps == 400 and bool(task) and
        metrics.get("initial_components") == 1 and metrics.get("final_components", 0) >= 2 and
        metrics.get("nonzero_digit_rewrites", -1) == 0 and metrics.get("reference_mass_absolute_error_kg", 1.) <= 1e-9 and
        metrics.get("postcut_persistence_time_s", 0.) >= .5 and
        all(score_checks.get(k) is True for k in ("complete_episode", "initial_continuous_graph", "initial_codes_zero",
             "reference_matches_single_continuous_task_sampler", "reference_mass_budget_and_unchanged_count",
             "persistent_history_no_digit_rewrites", "postcut_disconnected_topology_persists")))
    if task_id == "TRIM-01" and expected_steps == 400:
        physical &= score_checks.get("reference_above_plane_material_removed") is True
    failed = {k for k, v in score_checks.items() if v is not True}
    expected_trim = (task_id == "TRIM-01" and physical and score.get("status") == "FAIL_TASK_METRICS"
                     and bool(failed) and failed <= SURFACE_CHECKS)
    if expected_steps == 400:
        checks["independent_task_source_bytes"] = (task or {}).get("all_manifest_implementation_bytes_resolved") is True
        checks["recorded_task_recomputed"] = all((task or {}).get("recorded_score_recomputation", {}).get(k) is True
                                                for k in ("same_checks", "same_metrics", "same_status"))
        checks["finite_available_numerical_screens"] = all((task or {}).get("contact_and_numerical_screens", {}).get("checks", {}).values()) and bool(
            (task or {}).get("contact_and_numerical_screens", {}).get("checks"))
    return {"status": "PASS_REQUESTED_RUNTIME_CONTRACT" if all(checks.values()) else "FAIL_RUNTIME_CONTRACT",
        "checks": checks, "physical_cutting": "PASS_SAMPLED_SEPARATION" if physical else "FAIL" if expected_steps == 400 else "NOT_RUN_SMOKE",
        "strict_task_metrics": score.get("status", "NOT_RUN"), "expected_trim_surface_failure": bool(expected_trim),
        "camera_engineering_visibility": camera.get("visibility_and_optics", {}).get("status", "NOT_RUN"),
        "current_component_visibility": camera.get("current_component_visibility", "NOT_RUN"),
        "sensor_scope": camera.get("sensor_scope", {}), "full_benchmark_ready": False,
        "reset_clock_scope": "MEASURED_ZERO" if runtime == "isaac_sim" and checks["reset_clock_zero"] else "NOT_RECORDED_BY_EXISTING_LAB_RESET_RECEIPT" if runtime == "isaac_lab" else "FAIL_OR_MISSING",
        "camera_reset_repeatability": (reset or {}).get("camera_reset_repeatability", (reset or {}).get("depth_max_abs_error_m", {})),
        "reset_scope": "Measured physical state restoration; camera exact-mask/depth repeatability is separate"}


def summarize_validation(stages, *, tier, allow_expected_trim_geometry_fail=False):
    missing = not stages or any(s.get("status") in ("BLOCKED", "NOT_RUN", "FAIL") for s in stages)
    runtime = [s["classification"] for s in stages if s.get("kind") == "episode" and "classification" in s]
    runtime_ok = all(r["status"] == "PASS_REQUESTED_RUNTIME_CONTRACT" for r in runtime)
    full = [r for r in runtime if r["physical_cutting"] != "NOT_RUN_SMOKE"]
    physical_ok = all(r["physical_cutting"] == "PASS_SAMPLED_SEPARATION" for r in full)
    strict = all(r["strict_task_metrics"] == "PASS_TASK_METRICS" for r in full)
    acceptable = all(r["strict_task_metrics"] == "PASS_TASK_METRICS" or
                     (allow_expected_trim_geometry_fail and r["expected_trim_surface_failure"]) for r in full)
    static_ok = any(s.get("kind") == "static" and s.get("status") == "PASS" for s in stages)
    passed = static_ok and not missing and runtime_ok and physical_ok and acceptable and (tier == "static" or bool(runtime)) and (tier != "full" or bool(full))
    return {"status": ("PASS_DEPLOYMENT_WITH_EXPECTED_TRIM_SURFACE_FAILURE" if passed and not strict else
                       "PASS_REQUESTED_DEPLOYMENT_SCOPE" if passed else "FAIL_OR_BLOCKED_DEPLOYMENT_VALIDATION"),
            "requested_scope_pass": bool(passed), "tier": tier, "runtime_contracts_pass": bool(runtime_ok and runtime),
            "full_episode_physical_separation_pass": bool(physical_ok and full), "all_strict_full_task_metrics_pass": bool(strict and full),
            "allow_expected_trim_geometry_fail": allow_expected_trim_geometry_fail,
            "full_benchmark_ready": False, "hardware_sensor_calibration_validated": False,
            "scope": "Deployment execution, named RGB-D and sampled separation; strict geometry, native accuracy, contact, coupling-energy and convergence remain separate"}

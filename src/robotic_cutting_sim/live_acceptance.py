"""Evidence-gated analysis of recorded live Isaac cutting episodes.

No simulation runs here. PASS requires the measurements named by the frozen
protocol; missing controls remain NOT_RUN even when available metrics pass.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .cut_topology import mechanical_components
from .geometry import Specimen, sample_particles
from .recording import ROOT
from .scene_frames import (WorkpieceTarget, compose_pose, matrix_pose, pose_errors,
                           usd_camera_to_cv, usd_row_to_column)
from .scene_validation import (flattening_checks, legacy_heightfield, material_task_metrics,
                               mesh_heightfield_metrics, trajectory_metrics)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_recorded_source(directory, relative_path, expected_sha256):
    """Find hash-matching source without pretending a later copy existed at launch."""
    for candidate in (directory/"implementation"/Path(relative_path).name,
                      directory/"implementation"/relative_path, ROOT/relative_path):
        if candidate.is_file() and sha256(candidate) == expected_sha256:
            return {"verified": True, "source": str(candidate), "sha256": expected_sha256}
    history = subprocess.run(["git", "log", "--all", "--format=%H", "--", relative_path],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    for commit in history:
        blob = subprocess.run(["git", "show", commit+":"+relative_path], cwd=ROOT, capture_output=True)
        if blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == expected_sha256:
            return {"verified": True, "source": "git:"+commit+":"+relative_path,
                    "sha256": expected_sha256, "recovered_after_episode": True}
    return {"verified": False, "recorded_sha256": expected_sha256,
            "reason": "Matching implementation bytes were not found; the manifest hash is preserved"}


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def public_metrics(data):
    excluded = {"local_command_poses_xyzw", "local_actual_poses_xyzw", "position_error_m",
                "rotation_error_rad", "active_mask"}
    return {key: value for key, value in data.items() if key not in excluded and not key.endswith("_debug")}


def gate(checks=None, measurements=None, missing=(), scope=""):
    checks = {key: bool(value) for key, value in (checks or {}).items()}
    # A measured failure is not hidden by another missing measurement.
    status = "FAIL" if not all(checks.values()) else "NOT_RUN" if missing or not checks else "PASS"
    return {"status": status, "checks": checks, "measurements": measurements or {},
            "missing_evidence": list(missing), "scope": scope}


def same_partition(first, second):
    """Compare sets of connected particles independently of arbitrary label IDs."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 1:
        return False
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def cut_history_metrics(codes, cuts):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.dtype.kind not in "iu" or np.any(codes < 0) or np.any(codes >= 3**cuts):
        raise ValueError("Expected aligned recorded integer ternary cut history")
    rewrites = []
    for cut in range(cuts):
        digits = (codes//3**cut) % 3
        changed = (digits[:-1] != 0) & (digits[1:] != digits[:-1])
        rewrites.append(int(changed.sum()))
    return {"initial_nonzero_codes": int(np.count_nonzero(codes[0])),
            "nonzero_digit_rewrites_by_cut": rewrites,
            "nonzero_digit_rewrites": int(sum(rewrites)),
            "final_code_histogram": {str(code): int(count) for code, count in
                                     zip(*np.unique(codes[-1], return_counts=True))},
            "sampled_history_only": True}


def graph_audit(physics, cuts):
    names = ("reference_stencil_nodes", "reference_stencil_weights", "reference_stencil_gradients")
    if not all(key in physics for key in names):
        return {"status": "NOT_RUN", "reason": "Actual reference nodes, weights and stress gradients are required"}, None
    nodes, weights, gradients = [physics[key] for key in names]
    initial_count, initial = mechanical_components(nodes, weights, physics["codes"][0], cuts, gradients=gradients)
    final_count, final = mechanical_components(nodes, weights, physics["codes"][-1], cuts, gradients=gradients)
    recorded_matches = "component_ids" not in physics or same_partition(final, physics["component_ids"])
    return {"status": "PASS" if recorded_matches else "FAIL", "initial_components": initial_count,
            "final_components": final_count, "recorded_partition_matches_recomputation": recorded_matches,
            "includes_positive_weights": True, "includes_nonzero_stress_gradients": True,
            "includes_unassigned_code_zero_particles": True,
            "provenance": "Recomputed current compatible reference-node MPM graph from recorded actual nodes, weights, stress gradients and cut codes",
            "stencil_sha256": hashlib.sha256(nodes.tobytes()+weights.tobytes()+gradients.tobytes()).hexdigest()}, final


def matched_recordings(first_manifest, second_manifest, first_setup, second_setup,
                       first_robot, second_robot, *, feedback_control=False):
    """Check the recipe/path/source contract before interpreting a paired control."""
    a, b = dict(first_manifest.get("config", {})), dict(second_manifest.get("config", {}))
    # Output paths differ for independently exported but equivalent static scenes.
    for value in (a, b):
        value.pop("static_scene", None)
        value.pop("scene", None)
        if feedback_control:
            value.pop("feedback", None)
            # Camera readout does not advance or edit the material/robot. Its
            # disabled OFF control is explicitly recorded, never hidden.
            value.pop("camera", None)
    command = lambda robot: robot.get("desired_tool_pose_xyzw", robot.get("commanded_tool_pose_xyzw"))
    checks = {"same_config_except_declared_control": a == b,
              "same_material_recipe": first_setup.get("material") == second_setup.get("material"),
              "same_timestamps": np.array_equal(first_robot["time_s"], second_robot["time_s"]),
              "same_commanded_edge_path": np.array_equal(command(first_robot), command(second_robot))}
    if feedback_control:
        checks["feedback_toggled"] = first_manifest.get("config", {}).get("feedback") is True and second_manifest.get("config", {}).get("feedback") is False
    actual_runner = ("isaac_lab_env.py" if first_manifest.get("runtime", {}).get("source") == "live_isaac_lab"
                     else "isaac_sim_runner.py")
    for module in (actual_runner, "live_material.py", "tlmpm.py", "robot.py", "coupled_robot.py", "scene_frames.py"):
        key = "src/robotic_cutting_sim/"+module
        source_a = first_manifest.get("local_code_sha256", {}).get(key)
        source_b = second_manifest.get("local_code_sha256", {}).get(key)
        if source_a is not None or source_b is not None:
            checks["same_source_"+module] = source_a == source_b
    return checks


def repeated_state_metrics(first, second):
    if any(first[key].shape != second[key].shape for key in ("positions", "velocities", "codes")) or not np.array_equal(first["time_s"], second["time_s"]):
        return {"status": "NOT_RUN", "reason": "Repeat frames/particles/timestamps are not aligned"}
    mass = first["mass_kg"]
    result = {"all_frame_max_position_error_m": float(np.linalg.norm(first["positions"]-second["positions"], axis=-1).max()),
              "all_frame_max_velocity_error_m_s": float(np.linalg.norm(first["velocities"]-second["velocities"], axis=-1).max()),
              "positions_bitwise_equal": bool(np.array_equal(first["positions"], second["positions"])),
              "velocities_bitwise_equal": bool(np.array_equal(first["velocities"], second["velocities"])),
              "codes_bitwise_equal": bool(np.array_equal(first["codes"], second["codes"])),
              "differing_code_particle_frames": int(np.count_nonzero(first["codes"] != second["codes"])),
              "final_component_com_error_m": None}
    if "component_ids" in first and "component_ids" in second and same_partition(first["component_ids"], second["component_ids"]):
        com_errors = []
        for label in np.unique(first["component_ids"]):
            select = first["component_ids"] == label
            com = [np.average(state["positions"][-1, select], axis=0, weights=mass[select]) for state in (first, second)]
            com_errors.append(np.linalg.norm(com[0]-com[1]))
        result["final_component_com_error_m"] = float(max(com_errors))
    return result


def compare_standalone_reference(live_dir, reference_dir):
    """Describe an actual same-recipe cross-runtime comparison, never a replay pass."""
    live, reference = Path(live_dir), Path(reference_dir)
    setups = [_json(path/"setup.json") for path in (live, reference)]
    robots = [load_npz(path/"robot_state.npz") for path in (live, reference)]
    physics = [load_npz(path/"physics_state.npz") for path in (live, reference)]
    times_align = (robots[0]["time_s"].shape == robots[1]["time_s"].shape and
        np.allclose(robots[0]["time_s"], robots[1]["time_s"], atol=1e-9, rtol=0))
    effective_recipe = [dict(row["material"]) for row in setups]
    normalizations = []
    reference_manifest = _json(reference/"manifest.json")
    if "fixture_walls" not in effective_recipe[1] and effective_recipe[1].get("floor") is True:
        revision = reference_manifest.get("git_sha", "")
        original = subprocess.run(["git", "show", revision+":src/robotic_cutting_sim/tlmpm.py"], cwd=ROOT, capture_output=True)
        if original.returncode == 0 and b"for side in range(6):" in original.stdout and b"fixture_walls" not in original.stdout:
            effective_recipe[1]["fixture_walls"] = True
            normalizations.append({"field": "fixture_walls", "effective_value": True,
                "reason": "Recorded legacy floor=True unconditionally enabled all six finite tray/table contacts",
                "verified_source": "git:"+revision+":src/robotic_cutting_sim/tlmpm.py",
                "source_sha256": hashlib.sha256(original.stdout).hexdigest()})
    checks = {"identical_effective_material_recipe": effective_recipe[0] == effective_recipe[1],
        "control_timestamps_align": bool(times_align),
        "same_particle_reference_positions": np.array_equal(physics[0]["reference_positions"], physics[1]["reference_positions"]),
        "same_particle_masses": np.array_equal(physics[0]["mass_kg"], physics[1]["mass_kg"])}
    report = {"scope": "Measured standalone versus live-runtime difference, not a repeatability or Isaac runtime substitute",
        "reference": str(reference), "checks": checks,
        "raw_material_dicts_identical": setups[0]["material"] == setups[1]["material"],
        "recipe_schema_normalizations": normalizations,
        "live_contact_friction": setups[0]["material"]["contact_friction"],
        "reference_contact_friction": setups[1]["material"]["contact_friction"],
        "reference_input_sha256": {name: sha256(reference/name) for name in ("manifest.json", "setup.json", "robot_state.npz", "physics_state.npz")}}
    if times_align:
        command_error, command_angle = pose_errors(robots[0]["desired_tool_pose_xyzw"], robots[1]["desired_tool_pose_xyzw"])
        position_error, rotation_error = pose_errors(robots[0]["tool_pose_xyzw"], robots[1]["tool_pose_xyzw"])
        report.update({"max_commanded_edge_position_difference_m": float(command_error.max()),
            "max_commanded_edge_position_difference_after_initial_state_m": float(command_error[1:].max()),
            "max_commanded_edge_rotation_difference_rad": float(command_angle.max()),
            "max_actual_edge_position_difference_m": float(position_error.max()),
            "max_actual_edge_rotation_difference_rad": float(rotation_error.max()),
            "max_actual_joint_absolute_difference_rad": float(np.abs(robots[0]["q"]-robots[1]["q"]).max())})
    # Times differ only by the accumulated float64 clock representation here.
    aligned = dict(physics[1])
    if physics[0]["time_s"].shape == aligned["time_s"].shape and np.allclose(physics[0]["time_s"], aligned["time_s"], atol=1e-9, rtol=0):
        aligned["time_s"] = physics[0]["time_s"]
        report["material_state_difference"] = repeated_state_metrics(physics[0], aligned)
    return report


def actual_contact_kernel_recheck(physics, setup, component_ids, times_s, *, device="cpu", source_path=None):
    """Reevaluate actual contact kernel on saved states, without advancing any body.

    Component net impulse cancels all within-component antisymmetric pair forces.
    A net value above the reporting floor establishes cross-component exchange.
    It is not an original-substep pair log or an additional trajectory rollout.
    """
    import warp as wp
    if source_path is None:
        from . import tlmpm
    else:
        import importlib.util
        import sys
        name = "robotic_cutting_sim._recorded_contact_"+sha256(source_path)[:12]
        specification = importlib.util.spec_from_file_location(name, source_path)
        tlmpm = importlib.util.module_from_spec(specification)
        sys.modules[name] = tlmpm
        specification.loader.exec_module(tlmpm)
    config = setup["material"]
    radius = config["contact_radius_scale"]*config["dx"]*.25
    count = len(physics["mass_kg"])
    with wp.ScopedDevice(device):
        points = wp.empty(count, dtype=wp.vec3)
        velocity = wp.empty(count, dtype=wp.vec3)
        reference = wp.array(physics["reference_positions"], dtype=wp.vec3)
        mass = wp.array(physics["mass_kg"], dtype=float)
        codes = wp.empty(count, dtype=int)
        impulse = wp.zeros(count, dtype=wp.vec3)
        stats = wp.zeros(7, dtype=float)
        grid = wp.HashGrid(96, 64, 48)
        rows = []
        for index in times_s:
            points.assign(physics["positions"][index]); velocity.assign(physics["velocities"][index])
            codes.assign(physics["codes"][index]); impulse.zero_(); stats.zero_()
            grid.build(points, 2.4*radius)
            wp.launch(tlmpm._particle_contacts, count, [grid.id, points, reference, velocity, mass,
                codes, impulse, stats, radius, 3*config["dx"], config["dt"],
                config["particle_contact_stiffness"], config["contact_friction"]])
            measured = impulse.numpy().astype(float)
            by_component = [measured[component_ids == label].sum(0) for label in np.unique(component_ids)]
            rows.append({"time_s": float(physics["time_s"][index]), "component_net_impulses_Ns": by_component,
                         "total_impulse_budget_error_Ns": float(np.linalg.norm(measured.sum(0))),
                         "material_pair_events": int(stats.numpy()[4])})
    peak = max((np.linalg.norm(row["component_net_impulses_Ns"], axis=1).max() for row in rows), default=0.)
    return {"scope": "actual frozen contact kernel reevaluated on saved postwithdraw frames; not an original-substep pair log",
            "device": device, "peak_component_net_contact_impulse_Ns": float(peak),
            "source_path": None if source_path is None else str(source_path),
            "source_sha256": None if source_path is None else sha256(source_path),
            "roundoff_reporting_floor_Ns": 1e-9, "samples": rows}


def _json(path):
    return json.loads(Path(path).read_text())


def read_reset_metrics(directory):
    lab_path = directory/"reset_check.json"
    if lab_path.is_file():
        return {**_json(lab_path), "receipt_path": str(lab_path), "receipt_sha256": sha256(lab_path)}
    kit_path = directory/"reset_receipt.json"
    if not kit_path.is_file():
        return None
    original = _json(kit_path)
    result = {"scope": "same-process reset measured by native Kit runner" if original.get("dynamics_reset_measured") is True else "UNVERIFIED",
        "joint_max_abs_error_rad": original["q_max_error_rad"],
        "material_max_abs_error_m": original["material_max_error_m"], "nonzero_cut_codes": original["nonzero_cut_codes"],
        "depth_max_abs_error_m": {}, "depth_valid_mask_exact": {},
        "receipt_path": str(kit_path), "receipt_sha256": sha256(kit_path)}
    camera = original.get("camera_reset_repeatability")
    if isinstance(camera, dict):
        result["depth_max_abs_error_m"] = {name: row["max_depth_error_m"] for name, row in camera.items()}
        result["depth_valid_mask_exact"] = {name: row["valid_mask_changed_pixels"] == 0 for name, row in camera.items()}
        result["raw_camera_rechecks"] = {}
        for name in camera:
            initial_path = directory/("global_debug" if name == "global" else "observations")/f"{name}_0000.npz"
            reset_path = directory/"reset_observations"/(name+".npz")
            if not initial_path.is_file() or not reset_path.is_file():
                continue
            initial, reset = load_npz(initial_path), load_npz(reset_path)
            first, second = initial["depth_m"], reset["depth_m"]
            a, b = initial["valid_mask"].astype(bool), reset["valid_mask"].astype(bool)
            common = a & b
            delta = np.abs(first-second)
            measured_max = float(delta.max())
            changed = int(np.count_nonzero(a != b))
            result["raw_camera_rechecks"][name] = {"initial_sha256": sha256(initial_path), "reset_sha256": sha256(reset_path),
                "max_all_pixel_depth_error_m": measured_max, "valid_mask_changed_pixels": changed,
                "common_valid_p99_error_m": None if not common.any() else float(np.percentile(delta[common], 99)),
                "common_valid_max_error_m": None if not common.any() else float(delta[common].max()),
                "matches_instrumented_receipt": abs(measured_max-camera[name]["max_depth_error_m"]) <= 1e-12
                    and changed == camera[name]["valid_mask_changed_pixels"]}
            result["depth_max_abs_error_m"][name] = measured_max
            result["depth_valid_mask_exact"][name] = changed == 0
    return result


def native_camera_control_gates(directory, control_audit_path, manifest, protocol):
    """Bind independent noise-OFF controls to the exact live native sensor profile.

    These gates concern the native virtual RGB camera and SingleView source
    association. They do not claim the separate physical color-imager calibration.
    """
    from .single_view_registration import _require_controlled_optics_and_recipe
    audit = _json(control_audit_path)
    model = _json(directory/"native_depth_model_receipt.json")
    calibration = _json(directory/"calibration_runtime.json")
    frames = _json(directory/"frame_manifest.json")
    control_hashes = audit.get("input_sha256", {})
    hashes_match = bool(control_hashes) and all((ROOT/relative).is_file() and sha256(ROOT/relative) == expected
        for relative, expected in control_hashes.items())
    runtime = model["runtime"]
    runtime_match = (runtime.get("isaacsim_version") == "6.2.0.0" and
        runtime.get("kit_version") == "110.4.0+feature.386251.187cf21b.gl" and runtime.get("camera_extension_version") == "1.4.0")
    helper = "src/robotic_cutting_sim/single_view_registration.py"
    helper_matches = audit.get("implementation_sha256", {}).get(helper) == manifest.get("implementation_sha256", {}).get(helper)
    result = {}
    for number, camera in (("CAM-01", "d455_0"), ("CAM-02", "d455_1")):
        rows = [row for row in audit["rows"] if row["camera"] == camera and row["label"] == "off" and row["hypothesis"] == "H1"]
        limits = protocol["tests"][number]
        features = [row["feature"] for row in rows if row.get("feature") is not None]
        controls = {"known_plane_worst_p95_error_m": max((row["plane_error"]["p95_absolute_m"] for row in rows), default=None),
            "known_plane_minimum_valid_coverage": min((row["known_geometry_valid_coverage"] for row in rows), default=None),
            "registration_feature_worst_p95_error_px": max((row["boundary_error_p95_px"] for row in features), default=None),
            "noise_OFF_control_frames": len(rows), "development_and_heldout_splits": sorted({row["split"] for row in rows}),
            "control_audit_path": str(control_audit_path), "control_audit_sha256": sha256(control_audit_path),
            "control_protocol_sha256": audit["protocol_sha256"],
            "separate_physical_color_imager_calibration_validated": False,
            "full_dynamic_sensor_consistency": "Assessed independently under CAM-04"}
        checks = {"control_raw_input_hashes_match": hashes_match, "exact_runtime_profile": runtime_match,
                  "same_registration_implementation": helper_matches,
                  "both_development_and_heldout_controls": controls["development_and_heldout_splits"] == ["development", "held_out"],
                  "all_noise_OFF_planes_within_frozen_limit": bool(rows) and controls["known_plane_worst_p95_error_m"] <= limits["noise_off_plane_p95_absolute_error_m"],
                  "known_plane_valid_coverage": bool(rows) and controls["known_plane_minimum_valid_coverage"] >= limits["minimum_known_plane_valid_coverage"],
                  "feature_registration_error": bool(features) and controls["registration_feature_worst_p95_error_px"] <= limits["registration_feature_p95_error_px"],
                  "no_error_based_sample_rejection_in_controls": bool(rows) and all(row["registration"]["consistency_filter_applied"] is False for row in rows),
                  "all_native_values_exactly_preserved": bool(rows) and all(row["registration"]["native_measurement_values_preserved_exactly"] is True for row in rows)}
        try:
            _require_controlled_optics_and_recipe(calibration[camera], model["parameters"][camera])
            checks["exact_live_intrinsics_and_native_recipe"] = True
        except ValueError:
            checks["exact_live_intrinsics_and_native_recipe"] = False
        registrations = [row.get("native_registration", {}).get(camera, {}) for row in frames]
        checks["validated_profile_applied_to_every_live_frame"] = bool(registrations) and all(
            row.get("control_protocol_sha256") == audit["protocol_sha256"] and row.get("source_correspondence_validated") is True
            and row.get("consistency_filter_applied") is False and row.get("native_measurement_values_preserved_exactly") is True for row in registrations)
        result[number] = gate(checks, controls, scope="Actual Kit native virtual RGB/SingleView analytic geometry controls; no physical D455 color-imager or noise-realism claim")
    return result


def score_live_camera_records(records, protocol):
    """Keep a passing visual-geometry bridge separate from actual native holes."""
    cam3, cam4 = protocol["tests"]["CAM-03"], protocol["tests"]["CAM-04"]
    minimum_visible = {name: min(row["cameras"]["global"]["rtx_depth_consistent_attributed_pixels"][name] for row in records)
                       for name in cam3["required_objects"]}
    cameras = [row["cameras"][name] for row in records for name in ("d455_0", "d455_1", "global")]
    bridge = {"worst_p95_error_m": max(row["geometry_vs_actual_rtx"]["p95_absolute_error_m"] for row in cameras),
              "minimum_mask_iou": min(row["geometry_vs_actual_rtx"]["valid_mask_iou"] for row in cameras)}
    native = {name: {"worst_p95_error_m": max(row["cameras"][name]["actual_native_vs_geometry"]["p95_absolute_error_m"] for row in records),
                    "minimum_mask_iou": min(row["cameras"][name]["actual_native_vs_geometry"]["valid_mask_iou"] for row in records),
                    "maximum_absolute_error_m": max(row["cameras"][name]["actual_native_vs_geometry"]["max_absolute_error_m"] for row in records)}
              for name in ("d455_0", "d455_1", "global")}
    span = max(row["timestamp_span_s"] for row in cameras)
    chip = max(row["native_valid_chip_pixels"] for row in cameras)
    return {"CAM-03": gate({"all_required_objects_visible": all(value >= cam3["minimum_visible_pixels_each_required_object"] for value in minimum_visible.values())},
                {"minimum_RTX_depth_consistent_visible_pixels": minimum_visible, "actual_native_semantic_AOV": False},
                scope="Measured-state first-hit geometric attribution verified against actual native RTX depth, not a captured semantic AOV"),
            "CAM-04": gate({"physics_surface_camera_timestamps": span <= cam4["max_physics_surface_camera_timestamp_difference_s"],
                "visual_mesh_RTX_bridge_p95": bridge["worst_p95_error_m"] <= cam4["mesh_depth_p95_absolute_error_m"],
                "visual_mesh_RTX_bridge_masks": bridge["minimum_mask_iou"] >= cam4["minimum_reference_depth_valid_mask_iou"],
                "actual_native_depth_p95": all(row["worst_p95_error_m"] <= cam4["mesh_depth_p95_absolute_error_m"] for row in native.values()),
                "actual_native_valid_masks": all(row["minimum_mask_iou"] >= cam4["minimum_reference_depth_valid_mask_iou"] for row in native.values()),
                "actual_native_chip_visible": chip >= cam4["minimum_visible_chip_pixels_in_any_recorded_frame"]},
                {"geometry_bridge": bridge, "actual_native": native, "maximum_native_valid_chip_pixels": chip,
                 "maximum_timestamp_span_s": span, "error_tails_retained": True},
                scope="Actual native output must pass independently of the successful visual geometry bridge; no ideal/RTX fallback")}


def bound_live_camera_audit(directory, audit_path, protocol):
    audit = _json(audit_path)
    inputs = _json(audit_path.parent/"input.json")
    matches = Path(inputs["episode"]).resolve() == directory.resolve()
    for relative, expected in inputs["input_sha256"].items():
        path = ROOT/relative
        snapshot = audit_path.parent/"implementation"/Path(relative).name
        matches &= ((path.is_file() and sha256(path) == expected) or (snapshot.is_file() and sha256(snapshot) == expected))
    for row in audit["records"]:
        matches &= sha256(directory/"surface_debug"/f"surface_{row['frame']:04d}.npz") == row["surface_file_sha256"]
        for name, camera in row["cameras"].items():
            folder = "global_debug" if name == "global" else "observations"
            matches &= sha256(directory/folder/f"{name}_{row['frame']:04d}.npz") == camera["input_observation_sha256"]
    result = score_live_camera_records(audit["records"], protocol)
    for item in result.values():
        item["checks"]["recorded_live_evidence_hashes_match"] = bool(matches)
        item["checks"]["all_frames_audited_without_error_trimming"] = audit["partial_frame_subset"] is False and audit["all_valid_error_tails_retained"] is True
        item["measurements"]["independent_audit_path"] = str(audit_path)
        item["measurements"]["independent_audit_sha256"] = sha256(audit_path)
        if not all(item["checks"].values()):
            item["status"] = "FAIL"
    return result


def _usd_frame_audit(directory, setup, robot, target, diagnostics):
    path = directory/"final_live_scene.usda"
    native_kit = not path.is_file() and (directory/"calibration_runtime.json").is_file()
    if native_kit:
        path = Path(setup["source_scene"])
    if not path.is_file():
        return {}, ["Actual loaded/final USD scene export is absent"]
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(path))
        cache = UsdGeom.XformCache()
        measurements = {"up_axis": str(UsdGeom.GetStageUpAxis(stage)), "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                        "actual_final_scene_sha256": sha256(path), "camera_forward_cosines": {}, "camera_transform_errors": {}}
        point = target.points_to_world([0, 0, target.height_m])
        calibration_table = (_json(directory/"calibration_runtime.json") if native_kit else setup.get("calibration", {}))
        optical_paths = {"d455_0": "/World/D455_0/RSD455/Camera_Pseudo_Depth",
                         "d455_1": "/World/D455_1/RSD455/Camera_Pseudo_Depth", "global": "/World/GlobalCamera"}
        for name, calibration in calibration_table.items():
            prim = stage.GetPrimAtPath(calibration.get("prim_path", optical_paths[name]))
            world = usd_camera_to_cv(usd_row_to_column(np.array(cache.GetLocalToWorldTransform(prim)), measurements["meters_per_unit"]))
            expected = np.asarray(calibration["T_world_camera"])
            direction = point-expected[:3, 3]
            measurements["camera_forward_cosines"][name] = float(expected[:3, 2]@direction/np.linalg.norm(direction))
            pe, re = pose_errors(matrix_pose(expected), matrix_pose(world))
            measurements["camera_transform_errors"][name] = {"translation_m": float(pe), "rotation_rad": float(re)}
        if native_kit:
            attachment = [row["rendered_attachment"] for row in diagnostics]
            measurements["final_USD_vs_measured_hand_translation_m"] = max(row["translation_error_m"] for row in attachment)
            measurements["final_USD_vs_measured_hand_rotation_rad"] = max(row["rotation_error_rad"] for row in attachment)
            measurements["scope"] = "Native Replicator camera transforms versus loaded static USD; instrumented native PhysX versus live USD knife attachment at every control step"
        else:
            hand = stage.GetPrimAtPath(setup["cutting_hand_body_label"])
            actual_hand = matrix_pose(usd_row_to_column(np.array(cache.GetLocalToWorldTransform(hand)), measurements["meters_per_unit"]))
            pe, re = pose_errors(robot["hand_pose_xyzw"][-1], actual_hand)
            measurements["final_USD_vs_measured_hand_translation_m"] = float(pe)
            measurements["final_USD_vs_measured_hand_rotation_rad"] = float(re)
        return measurements, []
    except (ImportError, ValueError, KeyError, RuntimeError) as error:
        return {}, ["Actual USD transform check unavailable: "+str(error)]


def audit_live_episode(directory, protocol, *, protocol_path=None, contact_device=None,
                       feedback_off_dir=None, repeat_dir=None, native_camera_controls=None, live_camera_audit=None):
    directory = Path(directory)
    manifest, setup, completion = [_json(directory/name) for name in ("manifest.json", "setup.json", "result.json")]
    robot = load_npz(directory/"robot_state.npz")
    physics_path = directory/"physics_state.npz"
    if not physics_path.is_file():
        physics_path = directory/"oracle_debug/physics_state.npz"
    physics = load_npz(physics_path)
    diagnostic_path = directory/"diagnostics.json"
    if not diagnostic_path.is_file():
        diagnostic_path = directory/"oracle_debug/diagnostics.json"
    diagnostics = _json(diagnostic_path)
    trace = _json(directory/"trace.json") if (directory/"trace.json").is_file() else []
    if not trace and diagnostics and "minimum_detF" in diagnostics[0]:
        trace = [{"min_detF": row["minimum_detF"], "max_speed_m_s": row["maximum_material_speed_m_s"],
                  "time_s": row["timestamp_s"], "runtime_time_s": row["native_relative_timestamp_s"]} for row in diagnostics]
    target = WorkpieceTarget(tuple(setup.get("world_from_workpiece_xyzw", (0, 0, 0, 0, 0, 0, 1))),
        protocol["workpiece"]["target_plane_local_z_m"], protocol["workpiece"]["edge_clearance_above_target_m"])
    heightfield = legacy_heightfield(Specimen())
    runtime = manifest.get("runtime", {})
    source = runtime.get("source", completion.get("source", manifest.get("backend")))
    steps, duration = len(robot["time_s"])-1, float(robot["time_s"][-1]-robot["time_s"][0])
    gates = {name: gate(missing=["Required evidence has not been recorded"], scope=spec["reason"])
             for name, spec in protocol["tests"].items()}
    hashes = manifest.get("local_code_sha256", {})
    required_modules = ("isaac_lab_env.py", "live_material.py", "tlmpm.py", "cut_topology.py", "scene_frames.py")
    source_checks = {name: hashes.get("src/robotic_cutting_sim/"+name) == sha256(ROOT/"src/robotic_cutting_sim"/name)
                     for name in required_modules}
    resolved_sources = {name: resolve_recorded_source(directory, "src/robotic_cutting_sim/"+name,
        hashes.get("src/robotic_cutting_sim/"+name)) for name in required_modules} if source == "live_isaac_lab" else {}
    protocol_matches = protocol_path is None or manifest.get("acceptance_protocol_sha256") == sha256(protocol_path)
    if source == "live_isaac_lab":
        runtime_checks = {"full_400_step_recording": steps >= protocol["tests"]["RT-01"]["minimum_control_steps"],
            "full_8_second_recording": duration >= protocol["tests"]["RT-01"]["minimum_simulation_time_s"]-1e-9,
            "actual_scene_environment_task_api": completion.get("actual_scene_environment_task_api") is True,
            "actual_articulation_owner": runtime.get("articulation_owner") == "isaaclab.scene.InteractiveScene/NewtonManager",
            "actual_task_api": runtime.get("task_api") == "isaaclab.envs.DirectRLEnv",
            "GA_release": runtime.get("isaac_lab_release") == protocol["tests"]["RT-01"]["required_version"],
            "completed_execution": completion.get("status") == "PASS_EXECUTION",
            "protocol_matches_prefrozen_file": protocol_matches,
            "runtime_module_hashes_recorded": all(len(hashes.get("src/robotic_cutting_sim/"+name, "")) == 64 for name in required_modules)}
        if trace:
            runtime_checks["actual_native_clock_tracks_episode"] = max(abs(row["runtime_time_s"]-row["time_s"]) for row in trace) <= 1e-6
        gates["RT-01"] = gate(runtime_checks, {"steps": steps, "simulation_s": duration, "native_clock_s": completion.get("runtime_s"),
            "wall_s": completion.get("wall_s"), "current_source_matches_recorded_hash": source_checks,
            "hash_matching_implementation": resolved_sources,
            "GA_source_sha": runtime.get("isaac_lab_source_sha"),
            "distribution_versions": runtime.get("distribution_versions"),
            "package_version_is_not_GA_release_number": True}, missing=([] if trace else ["Native physics clock trace"])
            + ([] if all(row["verified"] for row in resolved_sources.values()) else ["Hash-matching recorded implementation source"]),
            scope="Live Lab DirectRLEnv + InteractiveScene/NewtonManager; kit-less cameras cannot pass native D455/Kit")
        gates["RT-02"] = gate(missing=["This recording is kit-less Lab, not a full Isaac Sim/Kit episode"])
    else:
        gates["RT-01"] = gate(missing=["This recording does not use the actual Lab task/scene APIs"])
        # A Kit adapter may provide all of these independent runtime receipts.
        if manifest.get("backend") == "LIVE_IsaacSim6.2_cutting":
            versions = manifest.get("versions", {})
            kit_sources = {name: resolve_recorded_source(directory, name, expected)
                for name, expected in manifest.get("implementation_sha256", {}).items()}
            gates["RT-02"] = gate({"full_400_steps": steps >= 400, "full_8_seconds": duration >= 8-1e-9,
                "native_physics_steps": completion.get("native_physics_steps", 0) >= 4000,
                "native_PhysX": completion.get("robot_dynamics_source") == "Isaac PhysX",
                "zero_standalone_robot_steps": completion.get("standalone_robot_dynamics_steps") == 0,
                "no_replay": completion.get("replay_inputs") is False,
                "live_captures": completion.get("camera_frames", 0) >= 2,
                "completed_execution": completion.get("status") == "PASS_EXECUTION",
                "version_6_2": any(str(v).startswith("6.2.0") for v in versions.values()),
                "native_process_exit_success": completion.get("process", {}).get("returncode") == 0,
                "frozen_implementation_bytes_verified": bool(kit_sources) and all(row["verified"] for row in kit_sources.values()),
                "actual_native_clock_tracks_episode": bool(trace) and max(abs(row["runtime_time_s"]-row["time_s"]) for row in trace) <= 1e-6},
                {"versions": versions, "steps": steps, "simulation_s": duration, "hash_matching_implementation": kit_sources}, scope="Actual Kit/PhysX/native capture path")

    usd, missing = _usd_frame_audit(directory, setup, robot, target, diagnostics)
    sc = protocol["tests"]["SC-01"]
    checks = {"identity_workpiece_supported_by_fixture": np.allclose(target.T_world_workpiece, np.eye(4), atol=1e-9, rtol=0)}
    checks["loaded_workpiece_local_target_matches_protocol"] = abs(setup.get("workpiece_local_target_height_m", target.height_m)-target.height_m) <= 1e-9
    if usd:
        checks.update({"USD_meters": usd["meters_per_unit"] == sc["usd_meters_per_unit"],
            "USD_Z_up": usd["up_axis"].upper() == sc["world_up_axis"],
            "all_required_camera_frames": set(usd["camera_forward_cosines"]) == {"global", "d455_0", "d455_1"},
            "camera_target_forward": all(value >= sc["minimum_camera_target_forward_cosine"] for value in usd["camera_forward_cosines"].values()),
            "camera_frame_translation": all(row["translation_m"] <= sc["max_transform_translation_error_m"] for row in usd["camera_transform_errors"].values()),
            "camera_frame_rotation": all(row["rotation_rad"] <= sc["max_transform_rotation_error_rad"] for row in usd["camera_transform_errors"].values()),
            "hand_USD_translation": usd["final_USD_vs_measured_hand_translation_m"] <= sc["max_transform_translation_error_m"],
            "hand_USD_rotation": usd["final_USD_vs_measured_hand_rotation_rad"] <= sc["max_transform_rotation_error_rad"]})
    gates["SC-01"] = gate(checks, usd, missing, "Actual USD versus recorded articulated hand and optical camera frames")

    cuts = setup["material"]["cuts"]
    history = cut_history_metrics(physics["codes"], cuts)
    graph, labels = graph_audit(physics, cuts)
    expected, expected_mass, _ = sample_particles(Specimen(), setup["material"]["dx"]/2, setup["material"]["density"])
    recorded_initial = target.points_to_local(physics["positions"][0])
    sampler_error = None if expected.shape != recorded_initial.shape else float(np.max(np.abs(expected-recorded_initial)))
    # This is a float representation bound, not a changed physical acceptance tolerance.
    sampler_roundoff = 2*float(np.max(np.abs(np.spacing(expected.astype(np.float32)))))
    gates["SC-02"] = gate({"actual_D0_samples": sampler_error is not None and sampler_error <= sampler_roundoff,
        "positive_masses": bool(np.all(physics["mass_kg"] > 0)), "initial_codes_zero": history["initial_nonzero_codes"] == 0,
        **({"initial_continuum_components_one": graph["initial_components"] == 1} if labels is not None else {})},
        {"initial_particle_count": len(physics["mass_kg"]), "initial_mass_kg": float(physics["mass_kg"].sum()),
         "actual_initial_sample_error_m": sampler_error, "float32_sample_roundoff_bound_m": sampler_roundoff,
         "analytic_specimen": {key: protocol["workpiece"][key] for key in ("length_m", "width_m", "base_height_m", "bump_height_m", "bump_radius_m")}},
        [] if labels is not None else [graph["reason"]], "Actual continuous initial D0 samples and full mechanical stencil graph")

    mount_limit = protocol["tests"]["ROB-01"]
    mount_metrics, mount_checks, mount_missing = {}, {}, []
    if "hand_pose_xyzw" in robot and "T_hand_edge_xyz_xyzw" in setup:
        pe, re = pose_errors(compose_pose(robot["hand_pose_xyzw"], setup["T_hand_edge_xyz_xyzw"]), robot["tool_pose_xyzw"])
        mount_metrics = {"max_fixed_mount_translation_error_m": float(pe.max()), "max_fixed_mount_rotation_error_rad": float(re.max()),
                         "named_hand": setup.get("cutting_hand_body_label"), "body_index": setup.get("cutting_hand_body_index")}
        articulations = setup.get("articulations", [{"joint_names": names} for names in setup.get("joint_names", [])])
        label = setup.get("cutting_hand_body_label", "")
        if "body_prim_paths" in robot and setup.get("cutting_hand_body_index") is not None:
            label = str(robot["body_prim_paths"][setup["cutting_hand_body_index"]])
        mount_checks = {"two_articulations": len(articulations) == 2,
            "seven_arm_joints_each": all(sum(name.startswith("panda_joint") for name in row["joint_names"]) == 7 for row in articulations),
            "named_actual_Panda_hand": "Franka_0/" in label and label.endswith("/panda_hand"),
            "physical_knife_attached_to_same_hand": setup.get("knife_shape_body_index", setup.get("physical_knife_body_index")) == setup.get("cutting_hand_body_index"),
            "mount_translation": float(pe.max()) <= mount_limit["max_fixed_mount_translation_error_m"],
            "mount_rotation": float(re.max()) <= mount_limit["max_fixed_mount_rotation_error_rad"]}
        if "body_pose" in robot and "cutting_hand_body_index" in setup:
            pe, re = pose_errors(robot["body_pose"][:, setup["cutting_hand_body_index"]], robot["hand_pose_xyzw"])
            mount_checks["recorded_hand_equals_named_body"] = float(pe.max()) <= 1e-5 and float(re.max()) <= 1e-4
    else:
        mount_missing.append("Measured hand trajectory and explicit fixed mount are required")
    gates["ROB-01"] = gate(mount_checks, mount_metrics, mount_missing, "Measured named hand and actual fixed physical wedge")

    command = robot.get("desired_tool_pose_xyzw", robot.get("commanded_tool_pose_xyzw"))
    trajectory = trajectory_metrics(robot["time_s"], command, robot["tool_pose_xyzw"], target, heightfield,
        protocol, setup["material"]["blade_half_width"])
    gates["ROB-02"] = gate(trajectory.get("checks"), public_metrics(trajectory),
        [] if trajectory["status"] != "NOT_RUN" else [trajectory["reason"]], "Measured edge geometry and tracking, not IK targets")
    stats = np.asarray([row["material_contact_statistics"] for row in diagnostics])
    contacts = int(stats[:, 1].sum())
    plastic = float(np.max(physics["final_plastic_strain"]))
    gates["PHY-01"] = gate({"actual_material_tool_contacts": contacts >= protocol["tests"]["PHY-01"]["minimum_material_tool_contact_events"],
        "nonzero_accumulated_plastic_history": plastic >= protocol["tests"]["PHY-01"]["minimum_max_accumulated_plastic_strain"]},
        {"material_tool_substep_particle_events": contacts, "max_accumulated_plastic_strain": plastic},
        ["Matched settle/elastic control measuring above-plane deformation residual drop"], "Constitutive deformation is separate from component separation")
    task = material_task_metrics(physics["positions"][0], physics["positions"][-1], physics["mass_kg"],
        labels if labels is not None else np.full(len(physics["mass_kg"]), -1), graph.get("provenance", ""), target, heightfield)
    checks = {"nonzero_digits_never_rewritten_in_recorded_history": history["nonzero_digit_rewrites"] == 0,
              "all_particle_frames_preserve_count": physics["positions"].shape[1] == len(physics["mass_kg"]),
              "initial_mass_matches_immutable_recorded_masses": abs(float(physics["mass_kg"].sum())-setup["initial_mass_kg"]) <= 1e-12}
    if labels is not None:
        checks.update({"initial_graph_one_component": graph["initial_components"] == 1,
            "final_graph_at_least_two": graph["final_components"] >= 2,
            "recorded_graph_agrees": graph["status"] == "PASS",
            "positive_detached_component_mass": task["detached_mass_kg"] >= protocol["tests"]["PHY-02"]["minimum_detached_mass_kg"]})
    gates["PHY-02"] = gate(checks, {"graph": graph, "history": history, "task_mass": public_metrics(task),
        "mass_evidence_scope": "All particle counts and one immutable per-particle mass vector; no inferred force observation"},
        [] if labels is not None else [graph["reason"]], "Persistent separation uses the actual transfer/stress stencil, including possible zero-code bridges")

    # The scripted blade is fully withdrawn at the end of the 0.8 s withdrawal.
    # Derive 6.4 s from the recorded cut stop plus the fixed published path phase.
    withdraw = protocol["episode"]["active_cut_interval_s"][1]+.8
    post = np.flatnonzero(physics["time_s"] >= withdraw-1e-9)
    chip_motion = None
    blade_clearance = None
    if labels is not None and task["status"] == "MEASURED" and len(post) >= 2:
        chip = ~task["substrate_mask_debug"]
        if chip.any():
            com = np.average(physics["positions"][post][:, chip], axis=1, weights=physics["mass_kg"][chip])
            chip_motion = float(np.linalg.norm(com-com[0], axis=1).max())
            from scipy.spatial.transform import Rotation
            clearances = []
            for frame in post:
                index = int(np.argmin(abs(robot["time_s"]-physics["time_s"][frame])))
                edge = robot["tool_pose_xyzw"][index]
                local = Rotation.from_quat(edge[3:]).inv().apply(physics["positions"][frame]-edge[:3])
                low = [-setup["material"]["blade_length"], -setup["material"]["blade_half_width"], -.0004]
                high = [0., setup["material"]["blade_half_width"], setup["material"]["blade_height"]]
                distance = np.linalg.norm(local-np.clip(local, low, high), axis=1)
                radius = setup["material"]["contact_radius_scale"]*setup["material"]["dx"]*.25
                clearances.append(distance.min()-radius)
            blade_clearance = float(min(clearances))
    contact = None
    contact_missing = []
    frozen_kernel = source_checks.get("tlmpm.py", False)
    frozen_source_path = None
    if not frozen_kernel:
        frozen_source_path = directory/"implementation/tlmpm.py"
        frozen_kernel = frozen_source_path.is_file() and sha256(frozen_source_path) == hashes.get("src/robotic_cutting_sim/tlmpm.py")
    if contact_device and labels is not None and len(post) and frozen_kernel:
        try:
            contact = actual_contact_kernel_recheck(physics, setup, labels, post, device=contact_device,
                source_path=frozen_source_path if frozen_source_path is not None and frozen_kernel else None)
        except Exception as error:
            contact_missing.append("Actual contact kernel recheck blocked: "+str(error))
    else:
        contact_missing.append("Actual saved-state contact kernel recheck (or original cross-component pair attribution)")
    phy3 = protocol["tests"]["PHY-03"]
    phy3_checks = {"postwithdraw_chip_motion": chip_motion is not None and chip_motion >= phy3["minimum_postwithdraw_chip_com_displacement_m"],
                   "persistent_history_no_reset": history["nonzero_digit_rewrites"] == 0,
                   "measured_blade_clear_of_all_particle_spheres_after_withdrawal": blade_clearance is not None and blade_clearance > 0}
    if contact is not None:
        phy3_checks["cross_component_contact_exchange"] = contact["peak_component_net_contact_impulse_Ns"] > max(phy3["minimum_cross_component_contact_impulse_Ns"], contact["roundoff_reporting_floor_Ns"])
    gates["PHY-03"] = gate(phy3_checks, {"withdrawal_complete_s": withdraw, "postwithdraw_chip_com_displacement_m": chip_motion,
        "postwithdraw_min_material_sphere_to_blade_AABB_clearance_m": blade_clearance,
        "original_postwithdraw_material_pair_events_unattributed": int(sum(row["material_contact_statistics"][4] for row in diagnostics if row["timestamp_s"] >= withdraw-1e-9)),
        "contact_kernel_recheck": contact}, contact_missing, "Independent chip motion and attributed component exchange; no healing")

    next_wrench = np.asarray([row["next_material_wrench_world"] for row in diagnostics])
    applied = np.asarray([row["applied_previous_material_wrench_world"] for row in diagnostics])
    carry = float(np.abs(applied[1:]-next_wrench[:-1]).max())
    material_work = float(stats[:, 2].sum())
    if all("applied_tool_wrench_work_j" in row for row in diagnostics):
        robot_work = float(sum(row["applied_tool_wrench_work_j"] for row in diagnostics))
        work_source = "Recorded actual applied-tool-wrench boundary-work ledger"
    else:
        from scipy.spatial.transform import Rotation
        edge = robot["tool_pose_xyzw"]
        rotation_delta = (Rotation.from_quat(edge[1:, 3:])*Rotation.from_quat(edge[:-1, 3:]).inv()).as_rotvec()
        robot_work = float(np.sum(applied[:, :3]*np.diff(edge[:, :3], axis=0))+np.sum(applied[:, 3:]*rotation_delta))
        work_source = "Recomputed from recorded actually applied wrenches and measured edge increments; world-angle increment, not desired motion"
    work_fraction = None if robot_work is None or abs(material_work) < 1e-12 else abs(material_work+robot_work)/abs(material_work)
    feedback = protocol["tests"]["PHY-04"]
    feedback_checks = {"finite_reaction": bool(np.isfinite(next_wrench).all()),
        "nonzero_reaction": float(np.linalg.norm(next_wrench[:, :3], axis=1).max()) >= feedback["minimum_peak_material_reaction_n"],
        "previous_interval_feedback_carry": carry <= feedback["max_previous_interval_feedback_carry_error_n"],
        "feedback_delay_declared": all(row["feedback_delay_s"] == feedback["declared_feedback_delay_s"] for row in diagnostics)}
    feedback_missing = []
    if work_fraction is None:
        feedback_missing.append("Material boundary and actual robot applied wrench work")
    else:
        feedback_checks["interface_work_residual"] = work_fraction <= feedback["max_interface_work_residual_fraction"]
    paired_feedback = None
    feedback_tool_difference = None
    if feedback_off_dir:
        off = Path(feedback_off_dir)
        paired_feedback = matched_recordings(manifest, _json(off/"manifest.json"), setup, _json(off/"setup.json"), robot, load_npz(off/"robot_state.npz"), feedback_control=True)
        feedback_checks.update(paired_feedback)
        if all(paired_feedback.values()):
            off_robot = load_npz(off/"robot_state.npz")
            difference = float(np.linalg.norm(robot["tool_pose_xyzw"][:, :3]-off_robot["tool_pose_xyzw"][:, :3], axis=1).max())
            feedback_tool_difference = difference
            feedback_checks["matched_feedback_affects_tool_dynamics"] = difference >= feedback["minimum_matched_on_off_tool_position_difference_m"]
    else:
        feedback_missing.append("Matched feedback ON/OFF live episode with identical recipe, code and command path")
    gates["PHY-04"] = gate(feedback_checks, {"peak_reaction_N": float(np.linalg.norm(next_wrench[:, :3], axis=1).max()),
        "feedback_carry_max_abs_error": carry, "material_boundary_work_J": material_work,
        "robot_applied_material_wrench_work_J": robot_work, "interface_work_residual_fraction": work_fraction,
        "robot_work_source": work_source,
        "matched_ON_OFF_max_tool_position_difference_m": feedback_tool_difference,
        "ON_camera_enabled": manifest.get("config", {}).get("camera"),
        "OFF_camera_enabled": None if not feedback_off_dir else _json(Path(feedback_off_dir)/"manifest.json").get("config", {}).get("camera"),
        "camera_control_scope": "Camera readout may be disabled in OFF; it does not advance or mutate dynamics",
        "scope": "Partitioned boundary work, not a total-energy convergence proof"}, feedback_missing)

    for number, camera in (("CAM-01", "d455_0"), ("CAM-02", "d455_1")):
        gates[number] = gate(missing=[f"Native {camera} noise-OFF plane/registration/coverage control tied to this sensor configuration"],
            measurements={"recorded_camera_source": runtime.get("camera"), "native_d455_validated": completion.get("native_d455_validated", False)},
            scope="Ideal Warp RGB-D and RTX reference cannot substitute for native D455")
    if native_camera_controls and manifest.get("backend") == "LIVE_IsaacSim6.2_cutting" and completion.get("camera_mode") == "native_single_view":
        gates.update(native_camera_control_gates(directory, native_camera_controls, manifest, protocol))
    frame_files = sorted((directory/"rgbd").glob("frame_*.npz"))
    timestamps = []
    global_rgb = 0
    for path in frame_files:
        frame = load_npz(path)
        if "timestamp_s" in frame and "surface_timestamp_s" in frame:
            timestamps.append(abs(float(frame["timestamp_s"])-float(frame["surface_timestamp_s"])))
        global_rgb += int("global_rgb" in frame)
    native_camera_timestamp_error = []
    if not frame_files and (directory/"frame_manifest.json").is_file():
        frame_manifest = _json(directory/"frame_manifest.json")
        for row in frame_manifest:
            index = row["frame"]
            for name, folder in (("global", "global_debug"), ("d455_0", "observations"), ("d455_1", "observations")):
                path = directory/folder/f"{name}_{index:04d}.npz"
                frame_files.append(path)
                frame = load_npz(path)
                timestamps.append(abs(float(frame["physics_timestamp_s"])-float(frame["surface_timestamp_s"])))
                global_rgb += int(name == "global" and "rgb" in frame)
                native_camera_timestamp_error.append(abs(float(frame["native_physics_timestamp_s"])-setup["native_time_origin_s"]-float(frame["physics_timestamp_s"])))
    gates["CAM-03"] = gate({"actual_live_RGB_frames_present": global_rgb >= 2}, {"global_RGB_frames": global_rgb},
        ["Per-object first-visible pixel attribution for both arms, knife, material and table"], "A complete scene view requires measured visibility, not prim existence")
    gates["CAM-04"] = gate({"recorded_physics_surface_camera_timestamps": bool(timestamps) and max(timestamps) <= protocol["tests"]["CAM-04"]["max_physics_surface_camera_timestamp_difference_s"]},
        {"RGBD_files": len(frame_files), "max_recorded_timestamp_difference_s": max(timestamps, default=None),
         "max_native_clock_offset_corrected_difference_s": max(native_camera_timestamp_error, default=None)},
        ["Independent full-scene mesh/depth agreement and valid-mask IoU", "Actual chip-visible pixel attribution"],
        "Recorded matching timestamps alone do not prove geometric capture agreement")
    if live_camera_audit:
        gates.update(bound_live_camera_audit(directory, Path(live_camera_audit), protocol))

    task1_checks = {} if task["status"] != "MEASURED" else {"detached_initial_above_target_mass_fraction":
        task["detached_initial_above_target_mass_fraction"] is not None and task["detached_initial_above_target_mass_fraction"] >= protocol["tests"]["TASK-01"]["minimum_detached_initial_above_target_mass_fraction"]}
    gates["TASK-01"] = gate(task1_checks, public_metrics(task), [] if task1_checks else ["Actual mechanical substrate/chip partition"],
        "Reference above-local-plane mass removed into physically disconnected components")
    surface_metrics = None
    final_step = int(round(float(robot["time_s"][-1])/protocol["episode"]["control_dt_s"]))
    surface_path = directory/"rgbd"/f"surface_{final_step:05d}.npz"
    metadata_path = surface_path.with_suffix(".json")
    metadata = _json(metadata_path) if metadata_path.is_file() else None
    if not surface_path.is_file() and (directory/"frame_manifest.json").is_file():
        metadata_path = directory/"frame_manifest.json"
        frame = max(_json(metadata_path), key=lambda row: row["time_s"])
        surface_path = directory/"surface_debug"/f"surface_{frame['frame']:04d}.npz"
        metadata = frame["current_component_surface"]
    if surface_path.is_file() and metadata is not None and labels is not None:
        surface = load_npz(surface_path)
        selected = [surface["triangles"][row["first_triangle"]:row["first_triangle"]+row["triangles"]]
                    for row in metadata["component_ranges"] if row["mechanical_component"] in task["reference_substrate_component_ids"]]
        surface_metrics = mesh_heightfield_metrics(surface.get("vertices", surface.get("vertices_m")), np.vstack(selected), target, heightfield,
            protocol["tests"]["TASK-02"]["surface_grid_spacing_m"])
        checks = flattening_checks(surface_metrics, protocol)["checks"]
        checks["no_future_partition"] = metadata.get("future_component_ids_used") is False
        checks["surface_partition_matches_actual_graph"] = metadata["mechanical_components"] == graph["final_components"]
        checks["surface_codes_match_physics"] = metadata.get("codes_sha256") == hashlib.sha256(physics["codes"][-1].tobytes()).hexdigest()
        checks["surface_stencil_matches_actual_graph"] = metadata.get("stencil_sha256") == graph["stencil_sha256"]
        checks["surface_and_physics_timestamp"] = abs(float(surface.get("timestamp_s", surface.get("physics_timestamp_s")))-float(physics["time_s"][-1])) <= 1e-6
        gates["TASK-02"] = gate(checks, {**public_metrics(surface_metrics), "source_mesh": str(surface_path),
            "source_mesh_sha256": sha256(surface_path), "source_metadata_sha256": sha256(metadata_path),
            "partition_rule": metadata["partition_rule"], "selection_scope": "Offline current mechanical-substrate triangle statistics; no physics or visibility edits"})

    repeat_measurements = None
    reset_missing = []
    reset_measurements = None
    repeat_checks = {}
    reset_measurements = read_reset_metrics(directory)
    if repeat_dir and (reset_measurements is None or not reset_measurements.get("depth_max_abs_error_m")):
        other_reset = read_reset_metrics(Path(repeat_dir))
        if other_reset is not None and other_reset.get("depth_max_abs_error_m"):
            reset_measurements = other_reset
    if reset_measurements is not None:
        limits = protocol["tests"]["ENV-01"]
        errors = reset_measurements.get("depth_max_abs_error_m", {})
        masks = reset_measurements.get("depth_valid_mask_exact", {})
        repeat_checks.update({"same_process_reset_after_cut": completion.get("steps", 0) >= 400 and history["final_code_histogram"] != {"0": len(physics["mass_kg"])}
                and "same-process" in reset_measurements.get("scope", ""),
            "reset_joint_state": reset_measurements["joint_max_abs_error_rad"] <= limits["initial_joint_max_abs_reset_error_rad"],
            "reset_material_state": reset_measurements["material_max_abs_error_m"] <= limits["initial_material_max_abs_reset_error_m"],
            "reset_all_cut_history_zero": reset_measurements["nonzero_cut_codes"] == 0,
            "reset_both_camera_depths": all(name in errors and errors[name] <= limits["initial_depth_max_abs_reset_error_m"] for name in ("d455_0", "d455_1")),
            "reset_both_camera_valid_masks": all(masks.get(name) is True for name in ("d455_0", "d455_1"))})
        if "raw_camera_rechecks" in reset_measurements:
            repeat_checks["camera_reset_receipt_matches_raw_recordings"] = all(row["matches_instrumented_receipt"] for row in reset_measurements["raw_camera_rechecks"].values())
    else:
        reset_missing.append("Reset after an actual terminal cut in the same environment, including q/material/depth/valid-mask/history comparisons")
    if repeat_dir:
        other = Path(repeat_dir)
        match = matched_recordings(manifest, _json(other/"manifest.json"), setup, _json(other/"setup.json"), robot, load_npz(other/"robot_state.npz"))
        repeat_checks.update(match)
        if all(match.values()):
            other_physics_path = other/"physics_state.npz"
            if not other_physics_path.is_file():
                other_physics_path = other/"oracle_debug/physics_state.npz"
            other_physics = load_npz(other_physics_path)
            repeat_measurements = repeated_state_metrics(physics, other_physics)
            limits = protocol["tests"]["ENV-01"]
            for measurement, limit in (("all_frame_max_position_error_m", "repeat_all_frame_max_position_error_m"),
                ("all_frame_max_velocity_error_m_s", "repeat_all_frame_max_velocity_error_m_s"),
                ("final_component_com_error_m", "repeat_final_component_com_error_m")):
                value = repeat_measurements.get(measurement)
                repeat_checks[measurement] = value is not None and value <= limits[limit]
    else:
        reset_missing.append("Independent full same-seed live episode with frozen recipe/commands")
    gates["ENV-01"] = gate(repeat_checks, {"repeat": repeat_measurements, "instrumented_same_process_reset": reset_measurements}, reset_missing,
        "Fresh-process similarity cannot alone prove clearing of a prior cut during reset")
    finite = all(np.isfinite(value).all() for data in (robot, physics) for value in data.values() if value.dtype.kind in "fc")
    env_limits = protocol["tests"]["ENV-02"]
    env_metrics = {"all_recorded_values_finite": finite,
        "minimum_detF": min((row["min_detF"] for row in trace), default=None),
        "maximum_material_speed_m_s": max((row["max_speed_m_s"] for row in trace), default=None),
        "maximum_retained_material_floor_tray_penetration_m": float(stats[:, 0].max()),
        "maximum_tool_material_penetration_m": float(stats[:, 3].max()),
        "maximum_material_overlap_m": float(stats[:, 6].max()),
        "sampling_scope": "Every control-boundary finite/detF/speed; retained material contacts use per-interval substep maxima; native Panda/self/table penetration is unmeasured"}
    env_checks = {"all_recorded_values_finite": finite,
        "retained_material_floor_tray_penetration": env_metrics["maximum_retained_material_floor_tray_penetration_m"] <= env_limits["maximum_non_tool_rigid_penetration_m"],
        "tool_material_penetration": env_metrics["maximum_tool_material_penetration_m"] <= env_limits["maximum_tool_material_penetration_m"],
        "material_overlap": env_metrics["maximum_material_overlap_m"] <= env_limits["maximum_material_overlap_m"]}
    if trace:
        env_checks["minimum_detF"] = env_metrics["minimum_detF"] >= env_limits["minimum_detF"]
        env_checks["maximum_speed"] = env_metrics["maximum_material_speed_m_s"] <= env_limits["maximum_material_speed_m_s"]
    env_missing = []
    if all("native_rigid_contact_audit" in row for row in diagnostics):
        native = [row["native_rigid_contact_audit"] for row in diagnostics]
        env_metrics["native_non_tool_rigid_max_penetration_m"] = max(row["max_non_tool_penetration_m"] for row in native)
        env_metrics["native_tool_rigid_max_penetration_m"] = max(row["max_tool_rigid_penetration_m"] for row in native)
        env_metrics["native_contact_sampling_source"] = sorted({row["source"] for row in native})
        env_checks["native_non_tool_rigid_penetration"] = env_metrics["native_non_tool_rigid_max_penetration_m"] <= env_limits["maximum_non_tool_rigid_penetration_m"]
        env_checks["native_tool_rigid_penetration"] = env_metrics["native_tool_rigid_max_penetration_m"] <= env_limits["maximum_tool_material_penetration_m"]
        env_metrics["sampling_scope"] = env_metrics["sampling_scope"].replace("; native Panda/self/table penetration is unmeasured", "")
        env_metrics["sampling_scope"] += "; actual native rigid contacts sampled before every2ms substep and after final substep"
    else:
        env_missing.append("Comprehensive native rigid-body (Panda self/robot-table/fixture) penetration audit")
    if not trace:
        env_missing.append("All control-boundary determinant/speed trace")
    gates["ENV-02"] = gate(env_checks, env_metrics, env_missing,
        "Measured finite/contact screen, not calibrated contact or a convergence proof")
    hashes = {str(path.relative_to(directory)): sha256(path) for path in (directory/"manifest.json", directory/"setup.json",
        directory/"result.json", directory/"robot_state.npz", physics_path, diagnostic_path)}
    result = {"scope": "Independent quantitative audit of this actual recorded episode", "recording": str(directory),
        "runtime_source": source, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": None if protocol_path is None else sha256(protocol_path), "input_sha256": hashes,
        "gates": gates, "gate_counts": {name: sum(row["status"] == name for row in gates.values()) for name in ("PASS", "FAIL", "NOT_RUN", "BLOCKED")},
        "all_18_gates_pass": all(row["status"] == "PASS" for row in gates.values()),
        "execution_pass": any(gates[name]["status"] == "PASS" for name in ("RT-01", "RT-02")),
        "physical_cutting_pass": gates["PHY-02"]["status"] == "PASS", "surface_flattening_pass": gates["TASK-02"]["status"] == "PASS",
        "native_camera_geometry_controls_pass": all(gates[name]["status"] == "PASS" for name in ("CAM-01", "CAM-02")),
        "native_d455_pass": all(gates[name]["status"] == "PASS" for name in ("CAM-01", "CAM-02", "CAM-04")),
        "full_benchmark_ready": False,
        "force_in_policy_observation": any(any(token in name.lower() for token in ("force", "wrench", "oracle", "component")) for name in setup.get("observation_fields", [])),
        "unvalidated_global_task_integration": "D0–D3 native sensor/task integration remains separate from this one D0 episode"}
    return result, {"trajectory": trajectory, "robot": robot, "target": target,
                    "heightfield": heightfield, "surface": surface_metrics}

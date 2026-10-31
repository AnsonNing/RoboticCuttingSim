"""WP-12 Phase 6: agentic-control readiness checks (read-only harness helpers; never used by a policy).

Nothing here changes physics, a cutting law, a goal evaluator or a threshold. The helpers read what the
``cutting_research_v1`` policy env, ``scripts/run_wp10_live.EpisodeRecorder`` and ``scripts/collect_dataset.run_episode``
already write, and add:

* ``LEAK_KEYS_WP12``: privileged names of the WP-12 surface, material and geometry layers, walked in deployable
  observations and policy-facing infos in addition to ``policy_env.PRIVILEGED_INFO_KEYS``;
* ``independent_through_cut_program``: an open-loop ``rcs_action_program_v1`` for a vertical sever built only from the
  public task description (``env.task_spec()``: scene config, goal and the world pose), never from the goal planner;
* ``post_reset_state`` / ``attachment_snapshot``: harness-side (privileged) receipts of what a reset left behind
  (labels, material, oracle history, contacts, cut-event recorder, surface topology) and of the knife/camera mounts;
* ``classify_episode``: the failure classification of a stored episode;
* ``validate_episode_dir``: the spec §6.4 completeness check of one stored episode directory.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

READINESS_VERSION = "agentic_readiness_v1"
EPISODE_VALIDATOR_VERSION = "wp12_episode_record_v1"

# Surface reconstruction (cut_aware_surface_v1 provenance rows, the cut_events_v1 log, reference-grid topology),
# material model parameters (materials_wp12_v1 receipts) and geometry-provider internals (heightfield_v2 / SDF-CSG
# lattice reports). None of them may appear in a deployable observation or a policy-facing info.
LEAK_KEYS_WP12 = frozenset({
    # surface reconstruction and the Hermite record
    "surface", "surface_provenance", "surface_selection", "surface_provenance_row", "hermite", "cut_events",
    "cut_events_log", "codes_sha256", "audit", "qef", "embedding", "regions", "reference_grid", "link_signature",
    "face_varying_normals_world", "cut_triangles", "mesh_shells", "lattice_pieces", "mechanical_components",
    "vertex_reference", "dirty_cells",
    # material model parameters and receipts
    "parameter_receipt", "material_receipt", "material_parameters", "density_kg_m3", "young_modulus_pa",
    "poisson_ratio", "yield_stress_pa", "fracture_energy_j_m2", "material_contact_statistics",
    "current_material_reaction_impulse_world", "applied_previous_material_wrench_world",
    "next_finger_wrenches_about_edge_world", "pending_wrench",
    # geometry providers and the sampled lattice
    "lattice", "lattice_report", "floating_particles", "bottom_layer_particles", "lattice_dims", "surface_noise",
    "protrusion_ownership", "reference_positions_local", "reference_stencil_nodes", "reference_stencil_weights",
})


def leak_paths(tree, keys=None, path=""):
    """Paths of every privileged name (``PRIVILEGED_INFO_KEYS`` | ``LEAK_KEYS_WP12`` by default) in a nested tree,
    skipping an explicit ``info["privileged"]`` channel."""
    from .policy_env import PRIVILEGED_INFO_KEYS
    keys = (PRIVILEGED_INFO_KEYS | LEAK_KEYS_WP12) if keys is None else keys
    found = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            if key == "privileged" and path == "":
                continue
            here = f"{path}/{key}"
            if key in keys:
                found.append(here)
            found += leak_paths(value, keys, here)
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            found += leak_paths(value, keys, f"{path}[{i}]")
    return found


def recorded_leak_walk(directory):
    """Post-hoc walk of a STORED episode: the policy-facing step infos in actions.jsonl (minus the documented
    privileged grasp_status that PolicyReadyEnv removes before returning info), the recorded deployable
    proprioception and the deployable export keys, with ``PRIVILEGED_INFO_KEYS | LEAK_KEYS_WP12``."""
    from .policy_env import PRIVILEGED_STEP_INFO
    d = Path(directory)
    info_paths, proprio_paths, rows = [], [], 0
    for line in (d/"actions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        info = {k: v for k, v in (row.get("info") or {}).items() if k not in PRIVILEGED_STEP_INFO}
        for p in leak_paths(info):
            if p not in info_paths:
                info_paths.append(p)
        for key in ("observation_proprio", "input_observation_proprio"):
            for p in leak_paths(row.get(key) or {}):
                if p not in proprio_paths:
                    proprio_paths.append(p)
    export = []
    if (d/"episode_export.npz").exists():
        from .policy_env import PRIVILEGED_INFO_KEYS
        keys = PRIVILEGED_INFO_KEYS | LEAK_KEYS_WP12
        with np.load(d/"episode_export.npz", allow_pickle=False) as z:
            export = [k for k in z.files if any(part in keys for part in k.split("__"))]
    return {"rows": rows, "info_paths": info_paths[:20], "proprio_paths": proprio_paths[:20], "export_keys": export,
            "status": "PASS" if not (info_paths or proprio_paths or export) else "FAIL",
            "scope": "stored policy-facing info, recorded proprioception and the deployable export"}


# ----------------------------------------------------------------------------- an independent controller

def _pose_world(world_from_workpiece_xyzw, position_local, rotation_local):
    pose = np.asarray(world_from_workpiece_xyzw, float)
    R = Rotation.from_quat(pose[3:])
    return np.r_[R.apply(position_local)+pose[:3], (R*rotation_local).as_quat()].tolist()


def independent_through_cut_program(spec, start_pose_world, *, plane_index=0, step_index=None, cut_index=None,
                                    clearance_m=.008, safe_height_m=.03, stop_above_support_m=.0005,
                                    approach_speed_m_s=.05, cut_speed_m_s=.01, hold_steps=15):
    """Open-loop ``rcs_action_program_v1`` for one vertical (normal along local x) sever plane of ``spec``.

    Inputs: ``spec = env.task_spec()`` (public scene/goal configs and the world_from_workpiece pose) and the
    deployable start edge pose (``observation["edge_command_world_xyzw"]``). It does not call the goal planner:
    the blade normal is the plane normal, the advance is local -z, the edge line is n x a; it lowers from
    ``safe_height_m`` above the workpiece top to ``clearance_m`` above it, cuts down at ``cut_speed_m_s`` to
    ``stop_above_support_m`` above the support top (z = 0 in the workpiece frame for a flat board), holds and
    withdraws. Only vertical planes are built (the controller is deliberately simple).
    """
    goal = spec["goal"]
    steps = goal["steps"] if goal["type"] == "sequence" else [goal]
    severs = [s for s in steps if s["type"] == "sever"]
    if step_index is not None:
        severs = [steps[step_index]]
    plane = severs[0]["planes"][plane_index]
    n = np.asarray(plane["normal_local"], float)
    if not np.allclose(np.abs(n), [1., 0., 0.], atol=1e-9):
        raise ValueError("independent_through_cut_program builds vertical planes with normal +-x only")
    a = np.array([0., 0., -1.])
    rotation = Rotation.from_matrix(np.column_stack([a, np.cross(n, a), n]))
    base = spec["scene"]["workpiece"]["base_box_m"]
    top = float(base["height_m"])+max([float(p.get("height_m", 0.)) for p in spec["scene"]["workpiece"]["protrusions"]]
                                      or [0.])
    x = float(plane["origin_local_m"][0])
    pose = spec["world_from_workpiece_xyzw"]
    above = _pose_world(pose, [x, 0., top+safe_height_m], rotation)
    start = _pose_world(pose, [x, 0., top+clearance_m], rotation)
    end = _pose_world(pose, [x, 0., stop_above_support_m], rotation)
    extra = {} if cut_index is None else {"cut_index": int(cut_index)}
    return {"schema": "rcs_action_program_v1", "program_id": f"independent_vertical_sever_x{x*1e3:+.1f}mm",
            "description": "Phase 6 independent controller: scene-config geometry only, no goal planner",
            "segments": [
                {"kind": "edge_line", "label": "MOVE_ABOVE", "from": list(map(float, start_pose_world)), "to": above,
                 "speed_m_s": approach_speed_m_s},
                {"kind": "edge_line", "label": "LOWER", "from": above, "to": start, "speed_m_s": approach_speed_m_s},
                {"kind": "edge_line", "label": "CUT", "from": start, "to": end, "speed_m_s": cut_speed_m_s,
                 "cut_enabled": True, **extra},
                {"kind": "edge_pose", "label": "HOLD", "target": end, "steps": int(hold_steps)},
                {"kind": "edge_line", "label": "WITHDRAW", "from": end, "to": above, "speed_m_s": approach_speed_m_s},
                {"kind": "hold_until_time_limit", "label": "HOLD_END"}]}


# ----------------------------------------------------------------------------- harness-side reset receipts

def _sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()


def post_reset_state(env, previous=None):
    """Privileged harness receipt right after ``env.reset()`` (never given to a policy).

    ``previous`` is the receipt of the previous reset (identity checks of the per-episode objects)."""
    backend, material = env.backend, env.backend.material
    state = material.history_state()
    codes = np.asarray(state["codes"])
    reference = np.asarray(material.world_points, float)
    out = {"time_s": float(env.time_s), "task_id": env.task.task_id, "particles": int(len(codes)),
           "labelled_particles": int(np.count_nonzero(codes)),
           "material_at_reference_max_m": float(np.abs(np.asarray(state["positions"])-reference).max()),
           "material_velocity_max_m_s": float(np.abs(np.asarray(state["velocities"])).max()),
           "oracle_history_samples": int(len(env.inner._history)),
           "pending_material_wrench_norm": float(np.linalg.norm(getattr(material, "pending_wrench", np.zeros(6)))),
           "material_object_id": id(material), "codes_sha256": _sha_array(codes)}
    recorder = getattr(material, "cut_event_recorder", None)
    if recorder is not None:
        counters = recorder.metadata()
        out["cut_events"] = {"log_entries": int(len(recorder.log())), "log_sha256": counters["log_sha256"],
                             "recorder_object_id": id(recorder)}
    surface = getattr(backend, "cut_aware_surface", None)
    if surface is not None:
        backend.surface_frame()   # the reset extraction (first row of this episode)
        rows = surface.rows
        out["surface"] = {"object_id": id(surface), "rows": len(rows), "first_update_mode": rows[0]["update"]["mode"],
                          "first_cut_triangles": int(rows[0]["mesh"]["cut_triangles"]),
                          "first_audit_ok": bool(rows[0]["audit"]["ok"]),
                          "first_mesh_shells": int(rows[0]["audit"]["mesh_shells"])}
    contacts = getattr(backend, "tool_contacts", None)
    if contacts is not None:
        out["tool_contacts_object_id"] = id(contacts)
    grasp = getattr(backend, "grasp_reset_receipt", None)
    if grasp is not None:
        out["grasp_reset_status"] = grasp.get("status")
    checks = {"labels_clean": out["labelled_particles"] == 0,
              "material_at_reference": out["material_at_reference_max_m"] <= 1e-9,
              "material_at_rest": out["material_velocity_max_m_s"] <= 1e-9,
              "oracle_history_single_sample": out["oracle_history_samples"] == 1,
              "no_pending_contact_wrench": out["pending_material_wrench_norm"] == 0.}
    if "cut_events" in out:
        checks["cut_event_log_empty"] = out["cut_events"]["log_entries"] == 0
    if "surface" in out:
        s = out["surface"]
        checks["surface_topology_fresh"] = (s["rows"] == 1 and s["first_update_mode"] == "full"
                                            and s["first_cut_triangles"] == 0 and s["first_audit_ok"])
    if previous is not None:
        checks["new_material_object"] = out["material_object_id"] != previous["material_object_id"]
        if "surface" in out and "surface" in previous:
            checks["new_surface_extractor"] = out["surface"]["object_id"] != previous["surface"]["object_id"]
        if "cut_events" in out and "cut_events" in previous:
            checks["new_cut_event_recorder"] = (out["cut_events"]["recorder_object_id"]
                                                != previous["cut_events"]["recorder_object_id"])
        if "tool_contacts_object_id" in out and "tool_contacts_object_id" in previous:
            checks["new_tool_contact_sensor"] = out["tool_contacts_object_id"] != previous["tool_contacts_object_id"]
    if "grasp_reset_status" in out:
        checks["grasp_restabilized"] = out["grasp_reset_status"] == "PASS_STABLE_GRASP"
    out["checks"] = checks
    out["failing"] = sorted(k for k, ok in checks.items() if not ok)
    return out


def _relative(a, b):
    """T_a^-1 T_b of two XYZW poses as (translation, rotation vector)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    Ra = Rotation.from_quat(a[3:])
    return Ra.inv().apply(b[:3]-a[:3]), (Ra.inv()*Rotation.from_quat(b[3:])).as_rotvec()


def attachment_snapshot(env):
    """Privileged knife-in-hand and wrist-camera mount receipt (live runtimes; kinematic has neither)."""
    backend = env.backend
    measured = backend.measured() if hasattr(backend, "measured") else backend.proprioception()
    out = {"time_s": float(env.time_s)}
    if "tool_body_pose_xyzw" in measured and "hand_pose" in measured:
        t, r = _relative(np.asarray(measured["hand_pose"], float).reshape(-1, 7)[0], measured["tool_body_pose_xyzw"])
        out["hand_from_tool"] = {"translation_m": t.tolist(), "rotation_vector_rad": r.tolist()}
    grasp = (getattr(backend, "last_diagnostics", {}) or {}).get("grasp_status")
    if isinstance(grasp, dict):
        out["grasp"] = {k: grasp.get(k) for k in ("status", "slip_since_settling_m", "rotation_since_settling_rad")}
    trace = getattr(backend, "camera_pose_trace", None) or []
    if trace:
        worst, mounts = {}, {}
        for sample in trace:
            for name, cam in sample["cameras"].items():
                d = cam["measured_vs_current_USD"]
                row = worst.setdefault(name, {"translation_error_m": 0., "rotation_error_rad": 0.})
                row["translation_error_m"] = max(row["translation_error_m"], float(d["translation_error_m"]))
                row["rotation_error_rad"] = max(row["rotation_error_rad"], float(d["rotation_error_rad"]))
                if cam.get("T_hand_camera") is not None:
                    mounts[name] = cam["T_hand_camera"]
        out["camera_samples"] = len(trace)
        out["camera_measured_vs_usd_max"] = worst
        out["camera_hand_mounts"] = mounts
    return out


def attachment_drift(snapshots):
    """Drift of the knife-in-hand pose and of the wrist-camera mounts over a list of ``attachment_snapshot``s."""
    tool = [s["hand_from_tool"] for s in snapshots if "hand_from_tool" in s]
    out = {}
    if tool:
        t0, r0 = np.asarray(tool[0]["translation_m"]), Rotation.from_rotvec(tool[0]["rotation_vector_rad"])
        out["tool_in_hand_translation_drift_max_m"] = float(max(np.linalg.norm(np.asarray(s["translation_m"])-t0)
                                                                for s in tool))
        out["tool_in_hand_rotation_drift_max_rad"] = float(max(
            (Rotation.from_rotvec(s["rotation_vector_rad"])*r0.inv()).magnitude() for s in tool))
    mounts = [s.get("camera_hand_mounts") for s in snapshots if s.get("camera_hand_mounts")]
    if mounts:
        first = mounts[0]
        out["camera_mount_max_abs_change"] = float(max(
            np.abs(np.asarray(m[name], float)-np.asarray(first[name], float)).max() for m in mounts for name in first))
        out["camera_measured_vs_usd_translation_max_m"] = float(max(
            row["translation_error_m"] for s in snapshots for row in (s.get("camera_measured_vs_usd_max") or {}).values()))
    return out


# ----------------------------------------------------------------------------- failure classification

FAILURE_CLASSES = ("TASK_SUCCESS", "EXPECTED_FAILURE", "OUTCOME_FAILURE", "INTEGRITY_FAILURE", "TERMINATED_GRASP",
                   "TERMINATED_IK", "NOT_EVALUATED", "EXECUTION_ERROR")


def classify_episode(meta):
    """(class, reason) of a stored episode's ``episode.json`` (collector meta); an expected failure is named."""
    if meta is None:
        return "EXECUTION_ERROR", "no episode record"
    result = meta.get("task_result") or {}
    verdict = meta.get("verdict")
    termination = meta.get("termination_reason")
    if termination in ("GRASP_LOST", "INVALID_CONTACT_DATA") or str(termination).startswith("GRASP"):
        return "TERMINATED_GRASP", str(termination)
    if termination == "IK_FAILURES":
        return "TERMINATED_IK", termination
    if verdict is None or verdict == "NOT_RUN":
        return "NOT_EVALUATED", str(result.get("reason") or result.get("missing_evidence") or "no verdict")
    if verdict == "FAIL_INTEGRITY":
        return "INTEGRITY_FAILURE", ",".join(k for k, v in ((result.get("integrity") or {}).get("checks") or {}).items()
                                              if not v) or "integrity"
    if verdict == "PASS":
        return "TASK_SUCCESS", "goal verdict PASS"
    outcome = result.get("outcome") or {}
    failed = [k for k, v in (outcome.get("checks") or {}).items() if not v]
    reason = f"verdict {verdict}; failed checks {failed}; task end {meta.get('task_end_reason')}"
    if meta.get("expect") is not None and meta.get("expect") == verdict:
        return "EXPECTED_FAILURE", reason
    return "OUTCOME_FAILURE", reason


# ----------------------------------------------------------------------------- spec 6.4 completeness

SPEC_6_4_ITEMS = ("full_configuration", "runtime_profile", "action_sequence", "camera_observations", "physical_state",
                  "cut_events", "contact_diagnostics", "surface_reconstruction_metadata", "evaluation",
                  "failure_classification", "randomization_seed", "code_commit", "timing_information")


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_episode_dir(directory, *, require_cameras=None, require_surface_provenance=None):
    """Spec §6.4 completeness of one stored episode (collector / Phase 6 layout). Read-only.

    Returns ``{"status": PASS|FAIL, "items": {item: {status, checks, note}}, "failing": [...]}``. An item that
    does not apply to the runtime (cameras on the kinematic backend; cut-aware provenance for the default
    particle surface) is ``NOT_APPLICABLE`` with the reason, unless ``require_*`` demands it."""
    d = Path(directory)
    items = {}

    def item(name, checks, note=None, status=None):
        failing = sorted(k for k, ok in checks.items() if not ok)
        items[name] = {"status": status or ("PASS" if not failing else "FAIL"), "checks": checks,
                       "failing": failing, **({"note": note} if note else {})}

    def exists(name):
        return (d/name).exists()
    meta = json.loads((d/"episode.json").read_text()) if exists("episode.json") else None
    manifest = json.loads((d/"manifest.json").read_text()) if exists("manifest.json") else None
    m = meta or {}
    item("full_configuration", {
        "manifest_json": manifest is not None and "config" in (manifest.get("config") or {}),
        "task_json": exists("task.json"), "material_json": exists("material.json"),
        "episode_scene_goal": bool(m.get("scene")) and bool(m.get("goal")),
        "configured_task_payload": bool(m.get("configured_task_payload")),
        "tool_profile": bool(m.get("tool_profile")), "preset": bool(m.get("preset"))})
    profile = m.get("runtime_profile")
    item("runtime_profile", {
        "runtime_profile_or_explicit_numerics": bool(profile) or bool(m.get("physics_version", {}).get("particle_spacing_m")),
        "acceptance_label_consistent": (profile is None and m.get("acceptance_valid") is None) or (
            profile is not None and m.get("acceptance_valid") == bool(profile.get("acceptance_valid"))),
        "physics_version": bool(m.get("physics_version"))},
        note=None if profile else "no runtime_profile declared: the explicit numerics are in physics_version")
    actions = _read_jsonl(d/"actions.jsonl") if exists("actions.jsonl") else []
    log = _read_jsonl(d/"action_log.jsonl") if exists("action_log.jsonl") else []
    item("action_sequence", {
        "actions_jsonl_nonempty": bool(actions), "action_log_nonempty": bool(log),
        "rows_match_recorded_steps": len(actions) == int(m.get("steps_recorded", -1)),
        "every_row_has_submitted_action": all("submitted" in r for r in actions),
        "status_per_step": all("status" in r for r in log)})
    calibration = json.loads((d/"frame_manifest.json").read_text())[0].get("camera_calibration") if exists(
        "frame_manifest.json") else {}
    cameras = sorted(calibration or {})
    live = m.get("runtime_source") not in (None, "kinematic_ideal_tracking")
    need_cameras = live if require_cameras is None else require_cameras
    if need_cameras:
        files = {name: sorted((d/"observations").glob(f"{name}_*.npz")) for name in cameras}
        item("camera_observations", {
            "cameras_calibrated": bool(cameras),
            "wrist_d405_x2_and_global_d455": sum("d405" in c.lower() or "wrist" in c.lower() for c in cameras) >= 2
            and any("d455" in c.lower() or "global" in c.lower() for c in cameras),
            "frames_per_camera": bool(files) and all(len(v) > 0 for v in files.values())},
            note=f"cameras {cameras}")
    else:
        item("camera_observations", {}, note="kinematic ideal-tracking backend renders no cameras",
             status="NOT_APPLICABLE")
    physics = {}
    if exists("physics_state.npz"):
        with np.load(d/"physics_state.npz", allow_pickle=False) as z:
            physics = {k: z[k].shape for k in z.files}
    item("physical_state", {
        "physics_state_npz": bool(physics),
        "positions_velocities_codes": all(k in physics for k in ("positions", "velocities", "codes", "time_s")),
        "reference_lattice_and_stencil": all(k in physics for k in ("reference_positions", "reference_stencil_nodes")),
        "robot_state": exists("robot_state.json") and exists("robot_state.npz")})
    surface_block = m.get("surface")
    has_log = exists("cut_events_log.npy")
    item("cut_events", {
        "label_history_frames": bool(physics) and physics.get("codes", (0,))[0] >= 2,
        "per_step_cut_flags": bool(actions) and all("cut_enabled" in (r.get("submitted") or {}) for r in actions),
        **({"cut_events_log_npy": has_log} if surface_block else {})},
        note="label history = codes in physics_state.npz at the recorded frames" + (
            "; Hermite cut-event log = cut_events_log.npy" if surface_block else ""))
    def contact_row(row):
        for key in ("material_diagnostics", "privileged_diagnostics"):
            block = row.get(key)
            if isinstance(block, dict) and ("material_contact_statistics" in block or "next_material_wrench_world" in block):
                return True
        return False
    item("contact_diagnostics", {
        "material_contact_rows": bool(actions) and all(contact_row(r) for r in actions
                                                       if (r.get("info") or {}).get("return_home_phase") is None),
        **({"grasp_status_rows": all(r.get("grasp_status") is not None for r in actions)} if live else {})})
    need_surface = bool(surface_block) if require_surface_provenance is None else require_surface_provenance
    if need_surface:
        provenance = json.loads((d/"surface_provenance.json").read_text()) if exists("surface_provenance.json") else {}
        rows = provenance.get("rows") or []
        sha_ok = None
        if has_log and provenance.get("cut_events", {}).get("log_sha256"):
            from .cut_events import _sha
            sha_ok = _sha(np.load(d/"cut_events_log.npy", allow_pickle=False)) == provenance["cut_events"]["log_sha256"]
        item("surface_reconstruction_metadata", {
            "provenance_file": bool(provenance),
            "profile_recorded": (provenance.get("selection") or {}).get("surface_profile") is not None,
            "reference_grid_topology": bool((provenance.get("selection") or {}).get("reference_grid")),
            "source_hashes": bool((provenance.get("selection") or {}).get("source_sha256")),
            "recorder_log_hash_matches_file": bool(sha_ok),
            "hermite_counts_per_extraction": bool(rows) and all("crossings_by_source" in r.get("hermite", {}) for r in rows),
            "embedding_per_extraction": bool(rows) and all("embedding" in r for r in rows),
            "audit_per_extraction": bool(rows) and all("ok" in r.get("audit", {}) for r in rows),
            "frames_map_to_extractions": bool(provenance.get("frames")) and all(
                f.get("codes_match_extraction") for f in provenance["frames"])})
    else:
        item("surface_reconstruction_metadata",
             {"surface_extraction_status_recorded": "surface_extraction" in (m.get("task_result") or {})
              or "surface_unmeshable" in (m.get("task_result") or {}) or m.get("task_result") is not None},
             note="particle_surface_v1 (default) writes no provenance file by design; the evaluation records the "
                  "surface extraction status")
    item("evaluation", {"result_json": exists("result.json"),
                        "task_result_verdict": (m.get("task_result") or {}).get("verdict") is not None,
                        "integrity_layer": "integrity" in (m.get("task_result") or {}) or
                        (m.get("task_result") or {}).get("verdict") == "NOT_RUN"})
    phase6 = json.loads((d/"phase6.json").read_text()) if exists("phase6.json") else {}
    cls = phase6.get("failure_classification") or {}
    item("failure_classification", {
        "class_recorded": cls.get("class") in FAILURE_CLASSES,
        "termination_reason_recorded": "termination_reason" in m and "episode_state_final" in m,
        "class_matches_record": cls.get("class") == classify_episode(meta)[0] if meta else False})
    item("randomization_seed", {"seed": isinstance(m.get("seed"), int), "seed_source": bool(m.get("seed_source"))})
    git = m.get("git") or {}
    item("code_commit", {"git_sha": bool(git.get("git_sha")), "git_dirty_flag": "git_dirty" in git,
                         "packages": bool(git.get("packages"))})
    item("timing_information", {"episode_wall_s": isinstance(m.get("wall_s"), (int, float)),
                                "evaluation_wall_s": isinstance(m.get("evaluation_wall_s"), (int, float)),
                                "reset_wall_s": isinstance((m.get("reset_info") or {}).get("reset_wall_s"), (int, float)),
                                "per_step_wall_s": bool(log) and all("wall_s" in r for r in log)})
    failing = sorted(name for name, row in items.items() if row["status"] == "FAIL")
    return {"version": EPISODE_VALIDATOR_VERSION, "episode_dir": str(d), "status": "PASS" if not failing else "FAIL",
            "items": items, "failing": failing, "spec": "WP12_SPECIFICATION.md §6.4",
            "not_applicable": sorted(name for name, row in items.items() if row["status"] == "NOT_APPLICABLE")}

"""Policy dataset format ``rcs_dataset_v1`` (WP-11 Phase G): action programs, episode export, reader, replay.

Layout of one dataset (``scripts/collect_dataset.py``)::

    <root>/dataset_manifest.json   version, command, git commit, runtime, collector config
    <root>/index.json              one row per episode (id, dir, seed, controller, outcome, sha256 of the export)
    <root>/reliability.json        multi-episode reliability checks and the POLICY_READY flag
    <root>/runtime/                the runtime's own receipts (Native kit log, grasp resets, ...)
    <root>/episode_###/            scripts/run_wp10_live.EpisodeRecorder layout (actions.jsonl, frame_manifest.json,
                                   observations/<camera>_####.npz, physics_state.npz, robot_state.json/.npz,
                                   task.json, material.json, result.json) plus
        episode.json               scene/workpiece/tool spec, seed, task result, home outcome, termination, physical
                                   diagnostics, frequencies, reset checks, memory, commit/physics version
        action_log.jsonl           the policy env's per-step action status log
        episode_export.npz         consolidated DEPLOYABLE arrays (actions, timestamps, robot state, camera RGB-D +
                                   calibration); privileged material history stays in physics_state.npz

Frequencies are recorded, never assumed: control is the task control rate
(50 Hz); camera content is stored at the rate actually recorded (every N
control steps) and the policy-visible camera refresh rate is reported separately.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

DATASET_VERSION = "rcs_dataset_v1"
ACTION_PROGRAM_SCHEMA = "rcs_action_program_v1"
EXPORT_NAME = "episode_export.npz"
REQUIRED_EPISODE_FILES = ("manifest.json", "actions.jsonl", "frame_manifest.json", "physics_state.npz",
                          "robot_state.json", "task.json", "material.json", "result.json", "episode.json",
                          "action_log.jsonl")
SEGMENT_KINDS = ("edge_pose", "edge_line", "edge_delta", "hold", "hold_until_time_limit")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ----------------------------------------------------------------------------- action programs

def load_action_program(source):
    program = json.loads(Path(source).read_text()) if not isinstance(source, dict) else source
    if program.get("schema") != ACTION_PROGRAM_SCHEMA or not isinstance(program.get("segments"), list):
        raise ValueError(f"Expected an {ACTION_PROGRAM_SCHEMA} document with a segment list")
    for i, segment in enumerate(program["segments"]):
        if segment.get("kind") not in SEGMENT_KINDS:
            raise ValueError(f"segment {i}: kind must be one of {SEGMENT_KINDS}")
    return program


def _unit_pose(pose, where):
    pose = np.asarray(pose, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1.) > 1e-6:
        raise ValueError(f"{where}: a WORLD edge pose is a finite [x, y, z, qx, qy, qz, qw] with a unit quaternion")
    return pose


def expand_action_program(program, control_dt_s, total_steps):
    """Per-step ``(label, action)`` list of an open-loop action program (no feedback, no planner).

    Segment kinds: ``edge_pose`` (repeat one absolute target ``steps`` times;
    the env's action limits bound each step), ``edge_line`` (absolute targets on
    the straight line ``from`` -> ``to`` at ``speed_m_s``, orientation slerped),
    ``edge_delta`` (repeat a delta), ``hold`` (zero world delta) and
    ``hold_until_time_limit`` (zero deltas to ``total_steps``).
    """
    program = load_action_program(program)
    rows = []
    for i, segment in enumerate(program["segments"]):
        kind, label = segment["kind"], segment.get("label", f"SEG{i}")
        cut = bool(segment.get("cut_enabled", False))
        extra = {"cut_index": int(segment["cut_index"])} if "cut_index" in segment else {}
        if kind == "edge_pose":
            target = _unit_pose(segment["target"], f"segment {i}")
            rows += [(label, {"type": "edge_pose", "target": target.tolist(), "cut_enabled": cut, **extra})
                     for _ in range(int(segment["steps"]))]
        elif kind == "edge_line":
            start, end = _unit_pose(segment["from"], f"segment {i} from"), _unit_pose(segment["to"], f"segment {i} to")
            distance = float(np.linalg.norm(end[:3]-start[:3]))
            speed = float(segment["speed_m_s"])
            if not speed > 0:
                raise ValueError(f"segment {i}: speed_m_s must be positive")
            n = max(1, int(np.ceil(distance/(speed*control_dt_s)-1e-9)))
            slerp = Slerp([0., 1.], Rotation.from_quat([start[3:], end[3:]]))
            for k in range(1, n+1):
                u = k/n
                q = slerp(u).as_quat()
                rows.append((label, {"type": "edge_pose", "target": np.r_[start[:3]+(end[:3]-start[:3])*u, q].tolist(),
                                     "cut_enabled": cut, **extra}))
        elif kind == "edge_delta":
            delta = np.asarray(segment["target"], float)
            if delta.shape != (6,) or not np.isfinite(delta).all():
                raise ValueError(f"segment {i}: edge_delta target is a finite 6-vector")
            rows += [(label, {"type": "edge_delta", "target": delta.tolist(), "frame": segment.get("frame", "world"),
                              "cut_enabled": cut, **extra}) for _ in range(int(segment["steps"]))]
        elif kind == "hold":
            rows += [(label, {"type": "edge_delta", "target": [0.]*6, "frame": "world", "cut_enabled": False})
                     for _ in range(int(segment["steps"]))]
        else:
            rows += [(label, {"type": "edge_delta", "target": [0.]*6, "frame": "world", "cut_enabled": False})
                     for _ in range(max(0, int(total_steps)-len(rows)))]
    if len(rows) > total_steps:
        raise ValueError(f"The action program has {len(rows)} steps; the episode allows {total_steps}")
    return rows


# ----------------------------------------------------------------------------- consolidated export

def _stack(rows, key, shape, dtype=float):
    out = np.full((len(rows), *shape), np.nan if np.dtype(dtype).kind == "f" else -1, dtype)
    for i, row in enumerate(rows):
        value = row.get(key)
        if value is not None:
            value = np.asarray(value, dtype)
            if value.shape == shape:
                out[i] = value
    return out


def export_episode(episode_dir):
    """episode_export.npz from the recorder files: deployable arrays only; returns (path, summary)."""
    episode_dir = Path(episode_dir)
    actions = read_jsonl(episode_dir/"actions.jsonl")
    frames = json.loads((episode_dir/"frame_manifest.json").read_text())
    arrays = {"export_version": np.asarray(DATASET_VERSION)}
    executed, infos = [], []
    for row in actions:
        info = row.get("info") or {}
        infos.append(info)
        executed.append(row.get("submitted") or {})
    n = len(actions)
    arrays["step_time_s"] = np.asarray([row["time_s"] for row in actions], float)
    def stamp(row, info):
        value = info.get("action_timestamp_s")
        value = row.get("input_timestamp_s") if value is None else value
        return np.nan if value is None else float(value)
    arrays["action_timestamp_s"] = np.asarray([stamp(row, info) for row, info in zip(actions, infos)], float)
    # Steps not issued through PolicyReadyEnv.step: the preflighted physical return-home joint plan
    # (RETURN_HOME_PLAN). Datasets written before this label used LEGACY_UNVALIDATED for the same rows.
    arrays["action_status"] = np.asarray([info.get("action_status", "RETURN_HOME_PLAN" if str(row.get("policy_phase", ""))
                                                   .startswith("RETURN_HOME") else "NOT_VALIDATED_BY_POLICY_ENV")
                                          for info, row in zip(infos, actions)], str)
    arrays["executed_kind"] = np.asarray([(info.get("action_executed") or {}).get("kind", "RETURN_HOME"
                                          if str(row.get("policy_phase", "")).startswith("RETURN_HOME") else "UNKNOWN")
                                          for info, row in zip(infos, actions)], str)
    arrays["executed_mode"] = np.asarray([str(e.get("mode")) for e in executed], str)
    cartesian = [e if e.get("mode") == "cartesian_edge_pose_world" else {} for e in executed]
    joints = [e if e.get("mode") == "joint_position_rad" else {} for e in executed]
    arrays["executed_edge_target_world_xyzw"] = _stack(cartesian, "target", (7,))
    arrays["executed_joint_target_rad"] = np.stack([np.asarray(e["target"], float).reshape(-1)[:7]
                                                    if e.get("target") is not None else np.full(7, np.nan)
                                                    for e in joints]) if n else np.zeros((0, 7))
    arrays["cut_enabled"] = np.asarray([bool(e.get("cut_enabled", False)) for e in executed], bool)
    arrays["cut_index"] = np.asarray([int(e["cut_index"]) if e.get("cut_index") is not None else -1 for e in executed], int)
    arrays["requested_action_json"] = np.asarray([json.dumps(info.get("action_requested", e)) for info, e in
                                                  zip(infos, executed)], str)
    arrays["policy_phase"] = np.asarray([str(row.get("policy_phase", "")) for row in actions], str)
    arrays["termination_reason"] = np.asarray([str(info.get("termination_reason")) for info in infos], str)
    proprio = [row.get("observation_proprio") or {} for row in actions]
    for key, shape in (("q", (2, 7)), ("qdot", (2, 7)), ("hand_pose", (7,)), ("finger_q", (2, 2)),
                       ("finger_qdot", (2, 2)), ("tool_pose", (7,))):
        if any(key in p for p in proprio):
            arrays[f"proprio__{key}"] = _stack(proprio, key, shape)
    arrays["reward"] = np.asarray([float(row.get("reward") or 0.) for row in actions], float)
    # Cameras: stack each camera's recorded frames (time from each frame's own physics timestamp).
    names = sorted({path.name.rsplit("_", 1)[0] for path in (episode_dir/"observations").glob("*.npz")
                    if not path.name.startswith(("overview_rgb", "closeup_rgb"))})
    camera_rates = {}
    for name in names:
        files = sorted((episode_dir/"observations").glob(f"{name}_[0-9][0-9][0-9][0-9].npz"))
        rgb, depth, valid, times, K, T = [], [], [], [], [], []
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                rgb.append(data["rgb"])
                depth.append(data["depth_m"])
                valid.append(data["valid_mask"] if "valid_mask" in data else np.isfinite(data["depth_m"]))
                times.append(float(data["physics_timestamp_s"]))
                K.append(data["K"] if "K" in data else np.full((3, 3), np.nan))
                T.append(data["T_world_camera"] if "T_world_camera" in data else np.full((4, 4), np.nan))
        arrays.update({f"camera__{name}__rgb": np.stack(rgb), f"camera__{name}__depth_m": np.stack(depth),
                       f"camera__{name}__valid_mask": np.stack(valid), f"camera__{name}__time_s": np.asarray(times),
                       f"camera__{name}__K": np.stack(K), f"camera__{name}__T_world_camera": np.stack(T)})
        dt = np.diff(times)
        camera_rates[name] = {"frames": len(times), "median_interval_s": float(np.median(dt)) if len(dt) else None,
                              "recorded_rate_hz": float(1./np.median(dt)) if len(dt) and np.median(dt) > 0 else None}
    arrays["frame_time_s"] = np.asarray([row["time_s"] for row in frames], float)
    path = episode_dir/EXPORT_NAME
    np.savez_compressed(path, **arrays)
    return path, {"steps": n, "cameras": camera_rates, "export_sha256": sha256_file(path), "keys": sorted(arrays)}


# ----------------------------------------------------------------------------- reader / validator

def load_index(root):
    root = Path(root)
    index = json.loads((root/"index.json").read_text())
    if index.get("version") != DATASET_VERSION:
        raise ValueError(f"{root}: not an {DATASET_VERSION} dataset")
    return index


def load_episode(root, episode):
    """Episode metadata, the export arrays and the action rows of one indexed episode (id or index)."""
    root = Path(root)
    index = load_index(root)
    rows = index["episodes"]
    row = rows[episode] if isinstance(episode, int) else next(r for r in rows if r["episode_id"] == episode)
    directory = root/row["dir"]
    meta = json.loads((directory/"episode.json").read_text())
    with np.load(directory/EXPORT_NAME, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return {"row": row, "dir": directory, "meta": meta, "arrays": arrays, "actions": read_jsonl(directory/"actions.jsonl")}


def validate_episode(root, episode):
    """Integrity checks of one stored episode; returns {status: PASS|FAIL, checks, failing}."""
    from .policy_env import PRIVILEGED_KEYS
    loaded = load_episode(root, episode)
    row, directory, meta, arrays, actions = (loaded[k] for k in ("row", "dir", "meta", "arrays", "actions"))
    checks = {}
    checks["required_files"] = all((directory/name).exists() for name in REQUIRED_EPISODE_FILES)
    checks["export_sha256_matches_index"] = sha256_file(directory/EXPORT_NAME) == row.get("export_sha256")
    dt = float(meta["control_dt_s"])
    times = arrays["step_time_s"]
    task_steps = arrays["executed_kind"] != "RETURN_HOME"
    checks["action_rows_match_steps"] = len(actions) == len(times) == int(meta["steps_recorded"])
    task_times = times[task_steps]
    checks["task_clock_uniform_50hz"] = bool(len(task_times) == 0 or (
        abs(task_times[0]-dt) < 1e-9 and np.allclose(np.diff(task_times), dt, atol=1e-9, rtol=0)))
    stamps = arrays["action_timestamp_s"][task_steps]
    checks["action_timestamps_precede_results"] = bool(np.allclose(task_times-stamps, dt, atol=1e-9, rtol=0)) \
        if len(task_times) else True
    checks["monotonic_time"] = bool(np.all(np.diff(times) > 0)) if len(times) > 1 else True
    statuses = set(arrays["action_status"].tolist())
    from .policy_env import ACTION_STATUSES
    checks["action_statuses_known"] = statuses <= set(ACTION_STATUSES) | {"RETURN_HOME_PLAN", "LEGACY_UNVALIDATED"}
    checks["no_privileged_keys_in_export"] = not [key for key in arrays
                                                  if any(part in PRIVILEGED_KEYS for part in key.split("__"))]
    frame_times = set(np.round(arrays["frame_time_s"], 9).tolist())
    step_times = set(np.round(np.r_[0., times], 9).tolist())
    checks["frames_at_step_times"] = frame_times <= step_times
    for key in arrays:
        if key.startswith("camera__") and key.endswith("__time_s"):
            checks[f"{key}_at_step_times"] = set(np.round(arrays[key], 9).tolist()) <= step_times
    checks["evaluation_present"] = meta.get("task_result") is not None and meta["task_result"].get("verdict") is not None
    checks["termination_recorded"] = "termination_reason" in meta and "episode_state_final" in meta
    with np.load(directory/"physics_state.npz", allow_pickle=False) as physics:
        recorded = np.asarray(physics["time_s"], float)
        checks["physics_state_at_frame_times"] = bool(np.allclose(recorded, arrays["frame_time_s"], atol=1e-9))
        checks["physics_final_codes_hash_matches"] = (
            hashlib.sha256(np.ascontiguousarray(physics["codes"][-1]).tobytes()).hexdigest()
            == meta.get("final_codes_sha256"))
    failing = sorted(name for name, ok in checks.items() if not ok)
    return {"episode_id": row["episode_id"], "status": "PASS" if not failing else "FAIL", "checks": checks,
            "failing": failing}


def analyze_episode(root, episode):
    """Compact per-episode analysis: status counts, phases, cut steps, camera rates, outcome."""
    loaded = load_episode(root, episode)
    arrays, meta = loaded["arrays"], loaded["meta"]
    status, counts = np.unique(arrays["action_status"], return_counts=True)
    phases, phase_counts = np.unique(arrays["policy_phase"], return_counts=True)
    cameras = {key.split("__")[1]: int(len(arrays[key])) for key in arrays
               if key.startswith("camera__") and key.endswith("__time_s")}
    return {"episode_id": loaded["row"]["episode_id"], "controller": meta.get("controller"), "seed": meta.get("seed"),
            "steps": int(len(arrays["step_time_s"])), "status_counts": dict(zip(status.tolist(), counts.tolist())),
            "phase_counts": dict(zip(phases.tolist(), phase_counts.tolist())),
            "armed_steps": int(arrays["cut_enabled"].sum()), "camera_frames": cameras,
            "frequencies": meta.get("frequencies"), "verdict": (meta.get("task_result") or {}).get("verdict"),
            "home_outcome": (meta.get("home_outcome") or {}).get("status"),
            "termination_reason": meta.get("termination_reason")}


def replay_kinematic(root, episode, *, device=None):
    """Re-execute a KINEMATIC episode's executed task actions open loop on a fresh kinematic env; compare codes.

    Native episodes cannot be replayed without Isaac Sim: returns NOT_APPLICABLE there.
    """
    loaded = load_episode(root, episode)
    meta, arrays = loaded["meta"], loaded["arrays"]
    if meta.get("runtime_source") != "kinematic_ideal_tracking":
        return {"status": "NOT_APPLICABLE", "reason": "Only kinematic episodes replay without Isaac Sim"}
    from .scene_config import task_from_payload
    from .kinematic_env import make_kinematic_env  # noqa: F401  (registered-task path unused here)
    from .scene_config import make_configured_kinematic_env
    task = task_from_payload(meta["configured_task_payload"])
    options = dict(meta["runtime_options"])
    if device is not None:
        options["device"] = device
    env = make_configured_kinematic_env(task, seed=int(meta["seed"]), action_limits=None, **options)
    try:
        env.reset(seed=int(meta["seed"]))
        mask = arrays["executed_kind"] != "RETURN_HOME"
        targets, cuts, indices = (arrays["executed_edge_target_world_xyzw"][mask], arrays["cut_enabled"][mask],
                                  arrays["cut_index"][mask])
        for target, cut, index in zip(targets, cuts, indices):
            action = {"mode": "cartesian_edge_pose_world", "target": target, "cut_enabled": bool(cut)}
            if index >= 0:
                action["cut_index"] = int(index)
            env.step(action)
        codes = np.asarray(env.backend.material.history_state()["codes"])
        digest = hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()
        with np.load(loaded["dir"]/"physics_state.npz", allow_pickle=False) as physics:
            recorded = np.asarray(physics["codes"][-1])
        mismatched = int(np.count_nonzero(codes != recorded)) if codes.shape == recorded.shape else None
        result = env.evaluate()
        return {"status": "PASS" if mismatched == 0 else "FAIL", "steps_replayed": int(mask.sum()),
                "final_codes_mismatched": mismatched, "final_codes_sha256": digest,
                "recorded_final_codes_sha256": meta.get("final_codes_sha256"),
                "replay_verdict": result.get("verdict"), "recorded_verdict": (meta.get("task_result") or {}).get("verdict"),
                "device": options.get("device"),
                "scope": "Open-loop replay of the executed absolute edge targets on the kinematic backend"}
    finally:
        env.close()


# ----------------------------------------------------------------------------- multi-episode reliability (Test F)

def _slope(values):
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return None
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.asarray(values), 1)[0])


def _pose_difference(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rotation = (Rotation.from_quat(a[3:])*Rotation.from_quat(b[3:]).inv()).magnitude()
    return float(np.linalg.norm(a[:3]-b[:3])), float(rotation)


def determinism_signature(meta):
    receipt = {k: v for k, v in (meta.get("controller_receipt") or {}).items()}
    return json.dumps({"controller": meta["controller"], "receipt": receipt, "seed": meta["seed"]}, sort_keys=True,
                      default=str)


def reliability_report(metas, config, runtime, *, validations=(), execution_failures=(), planned_episodes=None):
    """Per-episode and cross-episode Test F checks against ``phaseG_reliability_v1``; POLICY_READY iff all pass."""
    th, memory_cfg = config["reset"], config["memory"]
    live = runtime != "kinematic"
    validation = {v["episode_id"]: v for v in validations}
    reference = metas[0]["reset_checks"] if metas else None
    episodes = []
    for meta in metas:
        rc = meta["reset_checks"]
        translation, rotation = _pose_difference(rc["tool_pose_world_xyzw"], reference["tool_pose_world_xyzw"])
        checks = {
            "reset_time_zero": rc["time_zero"], "labels_clean": rc["labels_clean"],
            "material_at_reference": rc["material_at_reference_max_m"] <= th["material_at_reference_max_m"],
            "material_at_rest": rc["material_velocity_max_m_s"] <= th["material_velocity_max_m_s"],
            "start_codes_identical_to_episode_0": rc["material_codes_sha256"] == reference["material_codes_sha256"],
            "oracle_history_reset": rc["oracle_history_single_sample"],
            "lifecycle_reset": rc["lifecycle_ready_after_reset"], "ik_counter_reset": rc["ik_failures_zero"],
            "termination_latch_reset": rc["not_terminated"], "control_steps_reset": rc["control_steps_zero"],
            "action_log_reset": rc["action_log_empty"], "observation_clock_at_reset": rc["observation_timestamp_matches"],
            "tool_pose_repeats_episode_0": (translation <= th["tool_pose_repeat_translation_max_m"]
                                            and rotation <= th["tool_pose_repeat_rotation_max_rad"]),
            **({"tool_velocity_repeats_episode_0": (
                    abs(rc["tool_linear_speed_m_s"]-reference["tool_linear_speed_m_s"]) <= th["tool_linear_speed_repeat_max_m_s"]
                    and abs(rc["tool_angular_speed_rad_s"]-reference["tool_angular_speed_rad_s"])
                    <= th["tool_angular_speed_repeat_max_rad_s"])}
               if th.get("tool_velocity_rule") == "repeat_episode_0" else
               {"tool_at_rest_after_reset": (rc["tool_linear_speed_m_s"] <= th["tool_linear_speed_max_m_s"]
                                             and rc["tool_angular_speed_rad_s"] <= th["tool_angular_speed_max_rad_s"])}),
            "action_observation_clocks_consistent": meta["clock_errors"] == 0,
            "no_privileged_keys_in_observations": meta["observation_privileged_leaks"] == 0,
            "recorder_closed": bool(meta["recorder_closed"]),
            "evaluated": meta.get("verdict") is not None,
            "expected_outcome": meta.get("expect") is None or meta.get("verdict") == meta.get("expect"),
            "stored_episode_validates": validation.get(meta["episode_id"], {}).get("status") == "PASS",
        }
        if live:
            camera_every = int(meta["preset"]["camera_every"])
            checks.update(
                grasp_restabilized=rc["grasp_reset_status"] == th["grasp_reset_status_live"],
                camera_frame_current_at_reset=bool(rc["camera_timestamp_matches"]),
                camera_latest_frame_age_bounded=meta["camera_age_max_s"] <= (camera_every-1)*meta["control_dt_s"]+1e-9)
        failing = sorted(name for name, ok in checks.items() if not ok)
        episodes.append({"episode_id": meta["episode_id"], "label": meta["label"], "controller": meta["controller"],
                         "seed": meta["seed"], "verdict": meta.get("verdict"), "expect": meta.get("expect"),
                         "home_outcome": (meta.get("home_outcome") or {}).get("status"),
                         "termination_reason": meta.get("termination_reason"), "checks": checks, "failing": failing,
                         "tool_pose_repeat": {"translation_m": translation, "rotation_rad": rotation},
                         "memory_after": meta.get("memory_after"), "wall_s": meta.get("wall_s")})
    start = int(memory_cfg["fit_from_episode"])
    rss = [m["memory_after"]["host_rss_mib"] for m in metas][start:]
    gpu = [m["memory_after"]["gpu_used_mib_total"] for m in metas][start:]
    rss_slope, gpu_slope = _slope(rss), _slope(gpu)
    groups = {}
    for meta in metas:
        groups.setdefault(determinism_signature(meta), []).append(meta)
    exact = runtime == "kinematic" and all(m["runtime_options"].get("device") == "cpu" for m in metas)
    determinism = []
    for signature, members in groups.items():
        if len(members) < 2:
            continue
        codes = {m["final_codes_sha256"] for m in members}
        verdicts = {m.get("verdict") for m in members}
        determinism.append({"episodes": [m["episode_id"] for m in members], "controller": members[0]["controller"],
                            "seed": members[0]["seed"], "final_codes_identical": len(codes) == 1,
                            "verdict_identical": len(verdicts) == 1, "rule": "exact codes (CPU)" if exact else "verdict",
                            "pass": (len(codes) == 1) if exact else (len(verdicts) == 1)})
    minimum = int(config["minimum_consecutive_episodes"]["kinematic" if not live else "isaac_sim"])
    cross = {
        "consecutive_episodes_in_one_process": len(metas) >= minimum and not execution_failures
        and (planned_episodes is None or len(metas) == planned_episodes),
        "host_memory_no_leak": rss_slope is not None and rss_slope <= memory_cfg["host_rss_slope_max_mib_per_episode"],
        "gpu_memory_no_leak": (gpu_slope is None and all(v is None for v in gpu)) or (
            gpu_slope is not None and gpu_slope <= memory_cfg["gpu_slope_max_mib_per_episode"]),
        "same_seed_same_controller_repeatable": all(row["pass"] for row in determinism),
        "failed_episode_stored_and_usable": any(m.get("verdict") not in ("PASS", None) for m in metas) and all(
            validation.get(m["episode_id"], {}).get("status") == "PASS" for m in metas if m.get("verdict") != "PASS"),
    }
    failing = sorted([f"{row['episode_id']}:{name}" for row in episodes for name in row["failing"]]
                     + [name for name, ok in cross.items() if not ok])
    return {"version": config["version"], "runtime": runtime, "episodes": episodes, "cross_episode": cross,
            "memory": {"host_rss_mib_after_episode": [m["memory_after"]["host_rss_mib"] for m in metas],
                       "gpu_used_mib_after_episode": [m["memory_after"]["gpu_used_mib_total"] for m in metas],
                       "host_rss_slope_mib_per_episode": rss_slope, "gpu_slope_mib_per_episode": gpu_slope,
                       "fit_from_episode": start},
            "determinism": determinism, "execution_failures": list(execution_failures),
            "validations": list(validations), "failing": failing, "POLICY_READY": not failing,
            "scope": "POLICY_READY means the API/lifecycle/storage reliability checks of this run passed; "
                     "it is not a task-success or policy-quality claim"}

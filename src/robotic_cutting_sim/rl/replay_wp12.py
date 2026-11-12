"""Replay adapter for the WP-12 Native recording format (P1-5 T5.5; Demo 1 CUT, ``scripts/run_wp10_live.py``).

Layout of one WP-12 configured-task run (``<run>/episode_000`` and ``<run>/runtime``), read-only:

* ``episode_000/actions.jsonl.gz`` (gzipped in place after the WP-12 analysis; read streaming, the source is never
  rewritten; a plain ``actions.jsonl`` is accepted too): one row per control step with
  ``privileged_diagnostics.measured_tool_pose`` (the edge pose Native passed to ``LiveCuttingMaterial.advance``),
  ``active_cut`` / ``cut_index`` (resolved) and ``submitted.cut_enabled`` / ``cut_index`` (default 0);
* ``episode_000/robot_state.npz``: ``time_s [T+1]``, ``body_pose [T+1, B, 7]``, ``body_prim_paths [B]``;
* ``episode_000/material.json``: solver options and the reset obstacle boxes / recipients;
* ``episode_000/physics_state.npz``: the Native physical samples (``time_s``, ``positions``, ``velocities``,
  ``codes`` per frame plus the reference arrays the frozen goal evaluator reads);
* ``runtime/setup.json``: the runtime config (``configured_task`` payload string, control/material dt, seed,
  surface profile), ``physical_tool.tool_from_edge_xyz_xyzw``, ``physical_knife_body_index`` and
  ``home_reset.measured_after_pregrasp.tool_edge_pose_xyzw`` (the material reset edge);
* ``runtime/finger_geometry_00.json``: the finger pads and the reset ``boxes_edge`` (= ``configure_pad_obstacles``
  input).

Native call sequence per control step (``isaac_sim_runner.LiveIsaacSimCutting.step`` at ``a960dc6`` and now):
``update_pad_obstacles(_pad_boxes(actual tool pose))`` then ``advance(tool pose, t, cutting, cut_index)`` and
read-only diagnostics (``debug_state()`` extrema at ``a960dc6``, identical to ``numerical_diagnostics("snapshot")``
now). The adapter returns the solver-independent ``rl.replay.MeasuredTrajectory`` (same provenance rules as the
WP-17 adapter: RECORDED edge, cut flags, reset edge, reset boxes; RECONSTRUCTED per-step pad boxes with the
identical ``finger_geometry.pad_boxes_edge`` path, verified against the records it can be checked against).
"""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .replay import REPLAY_OBSTACLES, MeasuredTrajectory, _quat_angle, _sha256

RECORDING_FORMAT = "wp12_run_wp10_live_v1"


def _json(path):
    return json.loads(Path(path).read_text())


def _open_actions(episode):
    episode = Path(episode)
    plain, packed = episode / "actions.jsonl", episode / "actions.jsonl.gz"
    if plain.exists():
        return plain, plain.open("rt")
    if packed.exists():
        return packed, gzip.open(packed, "rt")
    raise FileNotFoundError(f"{episode}: no actions.jsonl(.gz)")


def default_runtime_dir(episode_dir):
    """``<run>/episode_000`` -> ``<run>/runtime``."""
    return Path(episode_dir).absolute().parent / "runtime"


def read_action_rows(episode_dir):
    """Per-step records the replay needs (streaming read; the source file is never written)."""
    path, stream = _open_actions(episode_dir)
    rows = []
    with stream:
        for line in stream:
            row = json.loads(line)
            diag, submitted = row["privileged_diagnostics"], row["submitted"]
            rows.append(dict(step=int(row["step"]), timestamp_s=float(diag["timestamp_s"]),
                             edge=np.asarray(diag["measured_tool_pose"], float), active=bool(diag["active_cut"]),
                             resolved=int(diag["cut_index"]), enabled=bool(submitted["cut_enabled"]),
                             index=int(submitted.get("cut_index", 0)), mode=submitted.get("mode"),
                             phase=row.get("policy_phase"),
                             total_impulse=np.asarray(diag["current_total_material_reaction_impulse_world"], float),
                             finger_wrenches=diag.get("next_finger_wrenches_about_edge_world", {}),
                             material_wall_s=diag.get("material_wall_s")))
    return path, rows


def extract_wp12_trajectory(episode_dir, runtime_dir=None, *, locator_root=None):
    """Read one recorded WP-12 Native episode into a ``MeasuredTrajectory`` (CPU only, no Warp)."""
    from ..finger_geometry import pad_boxes_edge
    from ..scene_frames import compose_pose

    episode = Path(episode_dir)
    runtime = Path(runtime_dir) if runtime_dir is not None else default_runtime_dir(episode)
    setup = _json(runtime / "setup.json")
    finger = _json(runtime / "finger_geometry_00.json")
    material = _json(episode / "material.json")
    config = setup["config"]
    if config.get("configured_task") is None:
        raise ValueError("Not a configured-task recording (setup.json config.configured_task is missing)")
    payload = json.loads(config["configured_task"])
    solver = payload["scene"]["solver"]
    if config.get("surface_profile") == "cut_aware_surface_v1":
        raise NotImplementedError("cut_aware_surface_v1 attaches the cut-event recorder; replay does not model it")
    task_json = episode / "task.json"
    if task_json.exists() and _json(task_json).get("task_id") != payload["task_id"]:
        raise ValueError("episode task.json and the runtime configured task disagree on the task id")
    control_dt = float(config["control_dt"])
    max_cuts = int(material["max_cuts"])
    options = dict(control_dt_s=control_dt, material_dt_s=float(config["material_dt"]), dx_m=float(config["dx"]),
                   substeps_per_control_step=int(round(control_dt / float(config["material_dt"]))),
                   seed=int(config["seed"]), feedback=bool(config["feedback"]),
                   v7_2_optimizations=bool(config.get("v7_2_optimizations", True)),
                   material_diagnostics=config.get("material_diagnostics", "snapshot"),
                   max_cuts=max_cuts, solver=solver["module"], cutting_law=int(solver["cutting_law"]),
                   particle_spacing_m=float(solver["particle_spacing_m"]),
                   slots_per_node=material.get("slots_per_node"), code_encoding=material.get("code_encoding"),
                   cut_events=None, surface_profile=config.get("surface_profile"),
                   replay_obstacles=REPLAY_OBSTACLES, recording_format=RECORDING_FORMAT)
    if material.get("solver") != options["solver"] or int(material.get("cutting_law")) != options["cutting_law"]:
        raise ValueError("material.json and the compiled scene disagree on solver or cutting law")
    if int(solver["max_cuts"]) != max_cuts:   # the configured path takes K from the scene (config.max_cuts unused)
        raise ValueError("material.json and the compiled scene disagree on max_cuts")

    tool_from_edge = np.asarray(setup["physical_tool"]["tool_from_edge_xyz_xyzw"], float)
    knife_index = int(setup["physical_knife_body_index"])
    reset_edge = np.asarray(setup["home_reset"]["measured_after_pregrasp"]["tool_edge_pose_xyzw"], float)
    reset_boxes = np.asarray(finger["boxes_edge"], np.float32)
    pads = finger["pads"]
    recipients = (None, *[pad["body_path"] for pad in pads])
    if list(recipients) != list(material.get("obstacle_reaction_recipients", recipients)):
        raise ValueError("Finger recipients differ between finger_geometry_00.json and material.json")

    actions_path, rows = read_action_rows(episode)
    T = len(rows)
    steps = np.asarray([r["step"] for r in rows])
    if not np.array_equal(steps, np.arange(1, T + 1)):
        raise ValueError("actions must hold consecutive control steps 1..T")
    time_s = np.asarray([r["timestamp_s"] for r in rows])
    edge = np.stack([r["edge"] for r in rows])
    active = np.asarray([r["active"] for r in rows])
    resolved = np.asarray([r["resolved"] for r in rows], np.int32)
    submitted_enabled = np.asarray([r["enabled"] for r in rows])
    submitted_index = np.asarray([r["index"] for r in rows], np.int32)

    with np.load(episode / "robot_state.npz", allow_pickle=False) as robot:
        body_pose = np.asarray(robot["body_pose"], float)
        body_paths = [str(path) for path in robot["body_prim_paths"]]
        robot_time = np.asarray(robot["time_s"], float)
    if body_pose.shape[0] != T + 1 or not np.allclose(robot_time, control_dt * np.arange(T + 1), atol=1e-9):
        raise ValueError("robot_state.npz must hold the reset state plus one body-pose row per control step")
    pad_index = {pad["body_path"]: body_paths.index(pad["body_path"]) for pad in pads}

    def boxes_at(row, edge_pose):
        measured = {path: body_pose[row, index] for path, index in pad_index.items()}
        return np.asarray([reset_boxes[0], *pad_boxes_edge(pads, measured, edge_pose)], np.float32)

    boxes = np.stack([boxes_at(t + 1, edge[t]) for t in range(T)])
    reset_rebuilt = boxes_at(0, reset_edge)
    knife_edge = compose_pose(body_pose[:, knife_index], tool_from_edge)
    phases = [r["phase"] for r in rows]
    checks = dict(
        steps=T,
        time_matches_control_boundaries=bool(np.allclose(time_s, control_dt * np.arange(1, T + 1), atol=1e-9)),
        knife_body_to_edge_max_position_diff_m=float(np.abs(knife_edge[1:, :3] - edge[:, :3]).max()),
        knife_body_to_edge_max_rotation_diff_rad=float(_quat_angle(knife_edge[1:, 3:], edge[:, 3:]).max()),
        reset_knife_body_to_edge_position_diff_m=float(np.abs(knife_edge[0, :3] - reset_edge[:3]).max()),
        reset_pad_boxes_rebuilt_max_abs_diff_m=float(np.abs(reset_rebuilt - reset_boxes).max()),
        material_json_obstacle_boxes_max_abs_diff_m=float(np.abs(np.asarray(material["obstacle_boxes_edge"],
                                                                            np.float32) - reset_boxes).max()),
        cut_flag_matches_command=bool(np.array_equal(active, submitted_enabled)),
        cut_digit_matches_command=bool(np.array_equal(np.where(submitted_enabled, submitted_index, -1), resolved)),
        cutting_steps=int(active.sum()),
        first_cutting_step=int(steps[active][0]) if active.any() else None,
        last_cutting_step=int(steps[active][-1]) if active.any() else None,
        submitted_modes=sorted({str(r["mode"]) for r in rows}),
        return_home_steps=int(sum(1 for p in phases if str(p).startswith("RETURN_HOME:"))),
        native_reaction_steps=int(sum(1 for r in rows if np.any(r["total_impulse"] != 0))),
        native_finger_wrench_steps=int(sum(1 for r in rows if any(np.any(np.asarray(v) != 0)
                                                                 for v in r["finger_wrenches"].values()))))
    checks["pad_boxes_verified"] = bool(checks["reset_pad_boxes_rebuilt_max_abs_diff_m"] == 0.0
                                        and checks["knife_body_to_edge_max_position_diff_m"] < 1e-12)
    if not checks["pad_boxes_verified"]:
        raise ValueError("Pad box reconstruction could not be verified; use handle_only_pads_unverified instead")
    if not (checks["cut_flag_matches_command"] and checks["cut_digit_matches_command"]):
        raise ValueError("Recorded resolved cut flags/digits differ from the submitted commands")

    def locator(path):
        path = Path(path).absolute()
        if locator_root is not None:
            try:
                return str(path.relative_to(Path(locator_root).absolute()))
            except ValueError:
                pass
        return path.name

    inputs = {f"episode/{actions_path.name}": actions_path}
    inputs.update({f"episode/{name}": episode / name for name in ("robot_state.npz", "material.json", "task.json")})
    inputs.update({f"runtime/{name}": runtime / name for name in ("setup.json", "finger_geometry_00.json")})
    source = dict(episode=locator(episode), runtime=locator(runtime), recording_format=RECORDING_FORMAT,
                  sha256={key: _sha256(path) for key, path in inputs.items() if path.exists()},
                  task_id=payload.get("task_id"))
    provenance = dict(
        edge_pose="RECORDED actions privileged_diagnostics.measured_tool_pose (the pose Native passed to "
                  "LiveCuttingMaterial.advance = compose_pose(tool body pose, tool_from_edge))",
        cutting_and_cut_index="RECORDED privileged_diagnostics.active_cut/cut_index (resolved digit), equal to "
                              "submitted.cut_enabled/cut_index (default 0)",
        reset_edge_pose="RECORDED runtime/setup.json home_reset.measured_after_pregrasp.tool_edge_pose_xyzw",
        reset_pad_boxes="RECORDED runtime/finger_geometry_00.json boxes_edge (= configure_pad_obstacles input)",
        pad_boxes="RECONSTRUCTED finger_geometry.pad_boxes_edge(pads, recorded finger body poses of the same "
                  "control step from robot_state.npz, recorded edge pose); handle box = recorded boxes_edge[0]",
        options="RECORDED runtime/setup.json config (configured_task payload), material.json; material_diagnostics "
                "'snapshot' when the config predates the key (a960dc6 read debug_state() extrema: same values)")
    return MeasuredTrajectory(task_payload=payload, options=options, time_s=time_s, edge_pose_world_xyzw=edge,
                              pad_boxes_edge=boxes, cutting=active, cut_index=submitted_index,
                              reset_edge_pose_world_xyzw=reset_edge, reset_pad_boxes_edge=reset_boxes,
                              pad_recipients=recipients, provenance=provenance, checks=checks,
                              source=source).validate()


def native_reactions(rows, recipients):
    """Recorded Native per-step total material reaction impulse [T,6] and finger wrenches [T,2,6]."""
    total = np.stack([r["total_impulse"] for r in rows])
    fingers = np.asarray([[r["finger_wrenches"].get(name, [0.] * 6) for name in recipients] for r in rows], float)
    return total, fingers


class Wp12NativeSamples:
    """Lazy read-only view of ``physics_state.npz`` (each array is decompressed once on first access)."""

    KEYS = ("time_s", "positions", "velocities", "codes", "mass_kg", "reference_positions",
            "reference_positions_local", "reference_indices", "reference_stencil_nodes",
            "reference_stencil_weights", "reference_stencil_gradients")

    def __init__(self, episode_dir):
        self.path = Path(episode_dir) / "physics_state.npz"
        self._npz = np.load(self.path, allow_pickle=False)
        self._cache = {}
        missing = [key for key in self.KEYS if key not in self._npz.files]
        if missing:
            raise ValueError(f"{self.path}: missing {missing}")

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = self._npz[key]
        return self._cache[key]

    def __contains__(self, key):
        return key in self._npz.files

    def keys(self):
        return list(self._npz.files)

    def close(self):
        self._npz.close()
        self._cache.clear()

    def checkpoint(self):
        times = np.asarray(self["time_s"], float)
        return dict(sample_count=int(len(times)), first_time_s=float(times[0]), last_time_s=float(times[-1]),
                    complete=bool(len(times) == len(self["codes"]) == len(self["positions"])))


def native_samples(episode_dir):
    """``(samples, checkpoint)`` with the same contract as ``rl.replay.native_samples``."""
    samples = Wp12NativeSamples(episode_dir)
    return samples, samples.checkpoint()


# -- source-equivalence audit of the v5 replay path (recording commit vs the current checkout)

V5_PATH_SOURCES = ("src/robotic_cutting_sim/tlmpm_v5.py", "src/robotic_cutting_sim/live_material.py",
                   "src/robotic_cutting_sim/cut_topology.py", "src/robotic_cutting_sim/fixture_contacts.py",
                   "src/robotic_cutting_sim/material_config.py", "src/robotic_cutting_sim/finger_geometry.py",
                   "src/robotic_cutting_sim/tool_profile.py", "src/robotic_cutting_sim/coupled_robot.py",
                   "src/robotic_cutting_sim/scene_frames.py", "src/robotic_cutting_sim/scene_config.py",
                   "src/robotic_cutting_sim/goals.py", "src/robotic_cutting_sim/goal_planner.py",
                   "src/robotic_cutting_sim/geometry.py", "src/robotic_cutting_sim/tasks.py",
                   "src/robotic_cutting_sim/flat_board_workcell.py", "src/robotic_cutting_sim/materials_wp12.py")


def _git(root, *args, binary=False):
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=not binary)
    return out.stdout if out.returncode == 0 else None


def source_equivalence_audit(recording_commit, root):
    """SHA256 of each v5-path source at ``recording_commit`` vs now, plus the diff stat of changed files and the
    Native pad / advance call lines of ``isaac_sim_runner.py`` at the recording commit and now."""
    root = Path(root)
    files = {}
    for rel in V5_PATH_SOURCES:
        old = _git(root, "show", f"{recording_commit}:{rel}", binary=True)
        now = (root / rel).read_bytes() if (root / rel).exists() else None
        files[rel] = dict(recording=None if old is None else hashlib.sha256(old).hexdigest(),
                          current=None if now is None else hashlib.sha256(now).hexdigest())
        files[rel]["identical"] = files[rel]["recording"] is not None and files[rel]["recording"] == files[rel]["current"]
    changed = [rel for rel, row in files.items() if not row["identical"]]
    stat = _git(root, "diff", "--stat", recording_commit, "HEAD", "--", *changed) if changed else ""
    runner_rel = "src/robotic_cutting_sim/isaac_sim_runner.py"

    def call_lines(text):
        keys = ("update_pad_obstacles(", "configure_pad_obstacles(", "material.advance(", "def _pad_boxes",
                "pad_boxes_edge(self.pad_geometry", "handle_box_edge(self.task.tool_profile)")
        return [line.strip() for line in (text or "").splitlines() if any(k in line for k in keys)]
    runner_old = _git(root, "show", f"{recording_commit}:{runner_rel}")
    runner_now = (root / runner_rel).read_text()
    return dict(recording_commit=recording_commit, files=files, changed_files=changed, diff_stat=stat,
                runner_pad_and_advance_lines=dict(recording=call_lines(runner_old), current=call_lines(runner_now)),
                runner_pad_semantics_equal=call_lines(runner_old) == call_lines(runner_now))

"""Measured-trajectory replay of a complete Native episode through the material only (decision D-05).

Source of truth: a WP-17 Native attempt directory (``.../env/recording/attempt_000``) and its sibling
``.../env/native`` directory. The Native runtime (``isaac_sim_runner.LiveIsaacSimCutting``) feeds the
material, at every 50 Hz control boundary and after the PhysX substeps of that interval:

* ``update_pad_obstacles(boxes)`` with ``[handle_box_edge(profile), *pad_boxes_edge(pads, pad_body_poses,
  edge)]`` (float32 ``[3, 2, 3]``: edge-frame centre and half extents of the handle and the two measured
  finger pads), and then
* ``advance(edge, time_s, cutting=command.cut_enabled, cut_index=command.cut_index)`` with
  ``edge = compose_pose(tool_body_pose, tool_from_edge)`` (the measured free-body knife pose composed with the
  fixed edge offset; world frame, XYZ + XYZW), followed by
* ``numerical_diagnostics(material_diagnostics)`` (read-only safety extrema).

``reset(initial_edge, seed)`` runs once after ``configure_pad_obstacles(initial_boxes, recipients)``.

What is RECORDED and used directly: the edge pose passed to ``advance`` (``privileged_diagnostics.
measured_tool_pose`` in ``actions.jsonl``), the resolved cut flag/digit (``active_cut``/``cut_index``), the
reset edge pose (``setup.json`` ``home_reset.measured_after_pregrasp.tool_edge_pose_xyzw``), the initial
obstacle boxes (``finger_geometry_00.json`` ``boxes_edge``), the compiled task payload and every solver
option (``setup.json`` config, ``material.json``). What is RECONSTRUCTED with the identical code path
(``finger_geometry.pad_boxes_edge``): the per-step finger pad boxes, from the recorded body poses of the two
finger bodies (``robot_state.npz``, read from PhysX after each control step) and the recorded edge pose. The
reconstruction is checked against the records it can be checked against (knife body -> edge pose, reset
boxes) and the residuals are reported in ``MeasuredTrajectory.checks``.

The trajectory object is solver independent (plain arrays + task payload + options): a batched solver can
consume the same object. Replay results are compared with the Native samples, never substituted for them.
No robot, IK, PhysX, camera or reaction feedback runs here (the reaction is computed but drives nothing).
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = "rcs_measured_trajectory_v1"
NATIVE_SOLVERS = ("tlmpm_v5", "tlmpm_v7", "tlmpm_v7_1", "tlmpm_v7_2")   # isaac_sim_runner's configured-task rule


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _quat_angle(a, b):
    """Rotation angle between XYZW quaternion arrays (sign invariant)."""
    dot = np.abs(np.sum(np.asarray(a, float) * np.asarray(b, float), axis=-1))
    return 2 * np.arccos(np.clip(dot, -1., 1.))


@dataclass
class MeasuredTrajectory:
    """Per-control-step material input of one recorded episode (T steps; step t ends at ``time_s[t]``)."""
    task_payload: dict
    options: dict
    time_s: np.ndarray                       # [T] float64
    edge_pose_world_xyzw: np.ndarray         # [T, 7] float64, passed to LiveCuttingMaterial.advance
    pad_boxes_edge: np.ndarray               # [T, 3, 2, 3] float32, passed to update_pad_obstacles
    cutting: np.ndarray                      # [T] bool
    cut_index: np.ndarray                    # [T] int32 command digit (used only when cutting)
    reset_edge_pose_world_xyzw: np.ndarray   # [7]
    reset_pad_boxes_edge: np.ndarray         # [3, 2, 3] float32
    pad_recipients: tuple                    # (None, left finger body, right finger body)
    provenance: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    @property
    def steps(self):
        return len(self.time_s)

    @property
    def resolved_cut(self):
        """The digit the material labels at each step (-1 = no cutting), as ``advance`` resolves it."""
        return np.where(self.cutting, self.cut_index, -1).astype(np.int32)

    def validate(self):
        T = self.steps
        shapes = dict(time_s=(T,), edge_pose_world_xyzw=(T, 7), pad_boxes_edge=(T, 3, 2, 3), cutting=(T,),
                      cut_index=(T,), reset_edge_pose_world_xyzw=(7,), reset_pad_boxes_edge=(3, 2, 3))
        for name, shape in shapes.items():
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} has shape {np.shape(getattr(self, name))}, expected {shape}")
        for name in ("time_s", "edge_pose_world_xyzw", "pad_boxes_edge", "reset_edge_pose_world_xyzw",
                     "reset_pad_boxes_edge"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} is not finite")
        if np.any(self.pad_boxes_edge[..., 1, :] <= 0) or np.any(self.reset_pad_boxes_edge[:, 1] <= 0):
            raise ValueError("Obstacle half extents must be positive")
        dt = float(self.options["control_dt_s"])
        if not np.allclose(self.time_s, dt * np.arange(1, T + 1), atol=1e-9, rtol=0):
            raise ValueError("Trajectory times must be consecutive control boundaries starting at one interval")
        if np.any(self.cut_index < 0) or np.any(self.cut_index >= int(self.options["max_cuts"])):
            raise ValueError("Command cut digits must lie inside the retained history")
        if len(self.pad_recipients) != 3 or self.pad_recipients[0] is not None:
            raise ValueError("Expected the handle (None) and two finger recipients")
        return self

    # -- solver-independent persistence (arrays in .npz, everything else in .json)
    ARRAYS = ("time_s", "edge_pose_world_xyzw", "pad_boxes_edge", "cutting", "cut_index",
              "reset_edge_pose_world_xyzw", "reset_pad_boxes_edge")

    def save(self, stem):
        stem = Path(stem)
        np.savez_compressed(stem.with_suffix(".npz"), **{name: getattr(self, name) for name in self.ARRAYS})
        meta = dict(schema=SCHEMA, task_payload=self.task_payload, options=self.options,
                    pad_recipients=list(self.pad_recipients), provenance=self.provenance, checks=self.checks,
                    source=self.source, steps=self.steps)
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True, allow_nan=False))
        return stem

    @classmethod
    def load(cls, stem):
        stem = Path(stem)
        meta = _json(stem.with_suffix(".json"))
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"Not a {SCHEMA} trajectory")
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as data:
            arrays = {name: data[name] for name in cls.ARRAYS}
        return cls(task_payload=meta["task_payload"], options=meta["options"],
                   pad_recipients=tuple(meta["pad_recipients"]), provenance=meta["provenance"],
                   checks=meta["checks"], source=meta["source"], **arrays).validate()


def default_native_dir(attempt_dir):
    """``.../env/recording/attempt_000`` -> ``.../env/native``."""
    return Path(attempt_dir).absolute().parents[1] / "native"


def extract_trajectory(attempt_dir, native_dir=None, *, locator_root=None):
    """Read one recorded Native attempt into a ``MeasuredTrajectory`` (CPU only, no Warp)."""
    from ..finger_geometry import pad_boxes_edge
    from ..scene_frames import compose_pose

    attempt = Path(attempt_dir)
    native = Path(native_dir) if native_dir is not None else default_native_dir(attempt)
    setup = _json(native / "setup.json")
    finger = _json(native / "finger_geometry_00.json")
    material = _json(attempt / "material.json")
    payload = _json(attempt / "compiled_task.json")
    config = setup["config"]
    solver = payload["scene"]["solver"]
    if config.get("surface_profile") == "cut_aware_surface_v1":
        raise NotImplementedError("cut_aware_surface_v1 attaches the cut-event recorder; replay does not model it")
    if config.get("configured_task") is not None and json.loads(config["configured_task"]) != payload:
        raise ValueError("compiled_task.json differs from the configured task the Native runtime compiled")
    control_dt = float(config["control_dt"])
    options = dict(control_dt_s=control_dt, material_dt_s=float(config["material_dt"]), dx_m=float(config["dx"]),
                   substeps_per_control_step=int(round(control_dt / float(config["material_dt"]))),
                   seed=int(config["seed"]), feedback=bool(config["feedback"]),
                   v7_2_optimizations=bool(config.get("v7_2_optimizations", True)),
                   material_diagnostics=config.get("material_diagnostics", "snapshot"),
                   max_cuts=int(config["max_cuts"]), solver=solver["module"], cutting_law=int(solver["cutting_law"]),
                   particle_spacing_m=float(solver["particle_spacing_m"]),
                   slots_per_node=material.get("slots_per_node"), code_encoding=material.get("code_encoding"),
                   cut_events=None, surface_profile=config.get("surface_profile"))
    if material.get("solver") != options["solver"] or int(material.get("cutting_law")) != options["cutting_law"]:
        raise ValueError("material.json and the compiled scene disagree on solver or cutting law")
    if int(material.get("max_cuts")) != options["max_cuts"]:
        raise ValueError("material.json and the Native config disagree on max_cuts")

    tool_from_edge = np.asarray(setup["physical_tool"]["tool_from_edge_xyz_xyzw"], float)
    knife_index = int(setup["physical_knife_body_index"])
    reset_edge = np.asarray(setup["home_reset"]["measured_after_pregrasp"]["tool_edge_pose_xyzw"], float)
    reset_boxes = np.asarray(finger["boxes_edge"], np.float32)
    pads = finger["pads"]
    recipients = (None, *[pad["body_path"] for pad in pads])
    if list(recipients) != list(material.get("obstacle_reaction_recipients", recipients)):
        raise ValueError("Finger recipients differ between finger_geometry_00.json and material.json")

    rows = []
    with (attempt / "actions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            diag, submitted = row["privileged_diagnostics"], row["submitted"]
            rows.append((int(row["step"]), float(diag["timestamp_s"]), np.asarray(diag["measured_tool_pose"], float),
                         bool(diag["active_cut"]), int(diag["cut_index"]), bool(submitted["cut_enabled"]),
                         int(submitted.get("cut_index", 0))))
    T = len(rows)
    steps = np.asarray([r[0] for r in rows])
    if not np.array_equal(steps, np.arange(1, T + 1)):
        raise ValueError("actions.jsonl must hold consecutive control steps 1..T")
    time_s = np.asarray([r[1] for r in rows])
    edge = np.stack([r[2] for r in rows])
    active = np.asarray([r[3] for r in rows])
    resolved = np.asarray([r[4] for r in rows], np.int32)
    submitted_enabled = np.asarray([r[5] for r in rows])
    submitted_index = np.asarray([r[6] for r in rows], np.int32)

    with np.load(attempt / "robot_state.npz", allow_pickle=False) as robot:
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
        last_cutting_step=int(steps[active][-1]) if active.any() else None)
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

    inputs = {f"recording/{name}": attempt / name for name in
              ("actions.jsonl", "robot_state.npz", "compiled_task.json", "material.json", "physics_stream.json")}
    inputs.update({f"native/{name}": native / name for name in ("setup.json", "finger_geometry_00.json")})
    source = dict(attempt=locator(attempt), native=locator(native),
                  sha256={key: _sha256(path) for key, path in inputs.items() if path.exists()},
                  task_id=payload.get("task_id"))
    provenance = dict(
        edge_pose="RECORDED actions.jsonl privileged_diagnostics.measured_tool_pose (the pose Native passed to "
                  "LiveCuttingMaterial.advance = compose_pose(tool body pose, tool_from_edge))",
        cutting_and_cut_index="RECORDED privileged_diagnostics.active_cut/cut_index (resolved digit), equal to "
                              "submitted.cut_enabled/cut_index",
        reset_edge_pose="RECORDED native/setup.json home_reset.measured_after_pregrasp.tool_edge_pose_xyzw",
        reset_pad_boxes="RECORDED native/finger_geometry_00.json boxes_edge (= configure_pad_obstacles input)",
        pad_boxes="RECONSTRUCTED finger_geometry.pad_boxes_edge(pads, recorded finger body poses of the same "
                  "control step from robot_state.npz, recorded edge pose); handle box = recorded boxes_edge[0]",
        options="RECORDED native/setup.json config, compiled_task.json scene.solver, material.json")
    return MeasuredTrajectory(task_payload=payload, options=options, time_s=time_s, edge_pose_world_xyzw=edge,
                              pad_boxes_edge=boxes, cutting=active, cut_index=submitted_index,
                              reset_edge_pose_world_xyzw=reset_edge, reset_pad_boxes_edge=reset_boxes,
                              pad_recipients=recipients, provenance=provenance, checks=checks,
                              source=source).validate()


# -- driving the existing single-env material (reference wrapper; CUDA graph path as in Native)

def build_material(trajectory, device):
    """A ``LiveCuttingMaterial`` constructed exactly as the Native configured-task path does (not yet reset)."""
    from ..coupled_robot import CouplingConfig
    from ..live_material import LiveCuttingMaterial
    from ..scene_config import task_from_payload

    o = trajectory.options
    task = task_from_payload(trajectory.task_payload)
    config = CouplingConfig(control_dt=o["control_dt_s"], material_dt=o["material_dt_s"], dx=o["dx_m"],
                            device=device, feedback=o["feedback"])
    solver = task.solver_module if task.solver_module in NATIVE_SOLVERS else None
    material = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law,
                                   solver=solver, v7_2_optimizations=o["v7_2_optimizations"])
    if solver != o["solver"] or material.cutting_law != o["cutting_law"] or material.max_cuts != o["max_cuts"]:
        raise ValueError("Rebuilt material differs from the recorded solver options")
    if material.slots_per_node != o["slots_per_node"]:
        raise ValueError("Rebuilt material slot capacity differs from the recording")
    material.configure_pad_obstacles(trajectory.reset_pad_boxes_edge, list(trajectory.pad_recipients))
    return material, task


class MaterialReplay:
    """Drive one material through a ``MeasuredTrajectory`` with the Native call sequence per step."""

    def __init__(self, trajectory, material):
        self.trajectory, self.material = trajectory, material
        self.step_index = 0

    def reset(self):
        start = time.perf_counter()
        info = self.material.reset(self.trajectory.reset_edge_pose_world_xyzw, self.trajectory.options["seed"])
        self.step_index = 0
        return time.perf_counter() - start, info

    def step(self):
        t, traj = self.step_index, self.trajectory
        self.material.update_pad_obstacles(traj.pad_boxes_edge[t])
        diagnostics = self.material.advance(traj.edge_pose_world_xyzw[t], float(traj.time_s[t]),
                                            cutting=bool(traj.cutting[t]), cut_index=int(traj.cut_index[t]))
        diagnostics.update(self.material.numerical_diagnostics(traj.options["material_diagnostics"]))
        if diagnostics["cut_index"] != int(traj.resolved_cut[t]):
            raise AssertionError("Material resolved a different cut digit than the recording")
        self.step_index += 1
        return diagnostics

    @property
    def time_s(self):
        return self.material.time_s


# -- comparison with the Native physical samples

def digit_counts(codes, max_cuts):
    """Labelled particle counts per packed cut digit (d_k = (code >> 2k) & 3; 1 negative, 2 positive)."""
    codes = np.asarray(codes, np.int64)
    out = {}
    for k in range(int(max_cuts)):
        digit = (codes >> (2 * k)) & 3
        out[str(k)] = dict(labelled=int(np.count_nonzero(digit)), negative=int(np.count_nonzero(digit == 1)),
                           positive=int(np.count_nonzero(digit == 2)))
    return out


def compare_sample(native, replay, max_cuts):
    """Per-sample position/velocity/code differences between a Native sample and the replay state."""
    dp = np.linalg.norm(np.asarray(replay["positions"], float) - np.asarray(native["positions"], float), axis=1)
    dv = np.linalg.norm(np.asarray(replay["velocities"], float) - np.asarray(native["velocities"], float), axis=1)
    codes_n, codes_r = np.asarray(native["codes"]), np.asarray(replay["codes"])
    counts_n, counts_r = digit_counts(codes_n, max_cuts), digit_counts(codes_r, max_cuts)
    return dict(max_position_diff_m=float(dp.max()), mean_position_diff_m=float(dp.mean()),
                rms_position_diff_m=float(np.sqrt(np.mean(dp ** 2))), p99_position_diff_m=float(np.percentile(dp, 99)),
                max_velocity_diff_m_s=float(dv.max()), mean_velocity_diff_m_s=float(dv.mean()),
                different_code_particles=int(np.count_nonzero(codes_n != codes_r)),
                labelled_count_native=counts_n, labelled_count_replay=counts_r,
                labelled_count_diff={k: counts_r[k]["labelled"] - counts_n[k]["labelled"] for k in counts_n})


def native_samples(attempt_dir):
    """Native streamed physical recording (read-only memory maps) and its checkpoint."""
    from ..recording import load_physics_recording
    data, checkpoint = load_physics_recording(Path(attempt_dir))
    return data, checkpoint

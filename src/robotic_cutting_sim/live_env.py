"""Shared research API over actual Isaac Lab and Isaac Sim cutting runtimes.

The wrapper has no physics, IK or renderer of its own. Native backends retain
ownership of those operations; oracle history is used only by evaluate().
"""
from contextlib import contextmanager
import copy
from dataclasses import asdict

import numpy as np

from .research_env import vision_observation
from .tasks import ACTION_MODES, make_task


class CuttingTaskEnv:
    """reset/step/observe/render/close/evaluate for both shared cutting tasks.

    A mapping action contains mode, target and optional cut_enabled. Cartesian
    targets are absolute WORLD knife-edge XYZ+XYZW; joint targets are seven
    active-arm positions in radians. None runs the scripted reference baseline.
    Explicit actions arm cutting independently of the reference-path schedule.
    The second Franka holds its observation posture. No force or oracle state
    enters observe(); Global RGB-D is an explicit opt-in.
    """

    def __init__(self, backend, task_id=None, *, seed=42, action_mode=ACTION_MODES[0],
                 observation_profile="vision_proprio", include_global_camera=False):
        if action_mode not in ACTION_MODES or observation_profile not in ("vision_proprio", "vision_only"):
            raise ValueError("Unsupported action mode or observation profile")
        task = getattr(backend, "task", None)
        self.task = task or make_task(task_id, seed=seed)
        self.backend, self.action_mode, self.observation_profile = backend, action_mode, observation_profile
        self.include_global_camera = bool(include_global_camera)
        self._closed, self._ready = False, False
        self._history, self._observation = [], None

    def reset(self, seed=None):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if seed is None:
            seed = getattr(self.task, "seed", 42)
        self.backend.reset(seed=int(seed))
        self.task = self.backend.task
        self._history, self._ready = [], True
        self._refresh()
        self._record_oracle()
        return self.observe(), {"task_id": self.task.task_id, "seed": int(seed),
            "runtime_source": self.runtime_source, "task_protocol_sha256": self.task.to_dict()["protocol_sha256"]}

    @property
    def runtime_source(self):
        return ("live_isaac_lab" if hasattr(self.backend, "measured") else "live_isaac_sim")

    @property
    def time_s(self):
        return float(self.backend.time_s)

    def _require_ready(self):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._ready:
            raise RuntimeError("Call reset on the environment first")

    def _refresh(self):
        if hasattr(self.backend, "measured"):
            self._frames, proprio = self.backend.frames, self.backend.measured()
        else:
            self._frames, _ = self.backend.capture()
            proprio = self.backend.proprioception()
        self._observation = vision_observation(self._frames, self.backend.calibration,
            self.time_s, self.observation_profile, proprio,
            include_global_camera=self.include_global_camera)

    def _record_oracle(self):
        state = self.backend.material.world_state()
        self._history.append({"time_s": self.time_s,
            **{name: state[name].copy() for name in ("positions", "velocities", "codes")}})

    def step(self, action=None):
        self._require_ready()
        if self.time_s >= self.task.duration_s-1e-9:
            raise RuntimeError("Episode ended; call reset")
        if action is not None:
            if not isinstance(action, dict):
                action = {"mode": self.action_mode, "target": action, "cut_enabled": True}
            action = self.task.validate_action(action)
        self.backend.step(action)
        self._refresh()
        self._record_oracle()
        # Goal success is deliberately evaluated separately from elapsed time.
        return self.observe(), 0., False, self.time_s >= self.task.duration_s-1e-9, {
            "task_id": self.task.task_id, "timestamp_s": self.time_s, "runtime_source": self.runtime_source}

    def observe(self):
        self._require_ready()
        return copy.deepcopy(self._observation)

    def render(self, camera="global", mode="rgb_array"):
        self._require_ready()
        if camera == "global" and "global_d455" in self._frames:
            camera = "global_d455"
        if camera not in self._frames or mode not in ("rgb_array", "depth_array"):
            raise ValueError("Choose an available camera and rgb_array/depth_array")
        return self._frames[camera]["rgb" if mode == "rgb_array" else "depth_m"].copy()

    def evaluate(self):
        """Independent oracle score; a completed episode alone never means success."""
        self._require_ready()
        if len(self._history) < 2:
            return {"task_id": self.task.task_id, "status": "NOT_RUN", "success": False,
                    "episode_complete": False, "reason": "No physical step recorded since reset",
                    "runtime_source": self.runtime_source}
        material = self.backend.material
        bed = material.bed
        if hasattr(self.backend, "surface_arrays"):
            vertices, triangles = self.backend.surface_arrays
            metadata = self.backend.surface_info
        else:
            vertices, triangles = self.backend.surface
            metadata = self.backend.surface_metadata
        face_components = np.full(len(triangles), -1, np.int32)
        for row in metadata["component_ranges"]:
            first = row["first_triangle"]
            face_components[first:first+row["triangles"]] = row["mechanical_component"]
        if np.any(face_components < 0):
            raise ValueError("Every scored triangle needs actual current mechanical provenance")
        result = self.task.evaluate_oracle({
            "time_s": [s["time_s"] for s in self._history],
            **{name: np.asarray([s[name] for s in self._history]) for name in ("positions", "codes")},
            "mass_kg": material.masses, "reference_positions": material.world_points,
            "reference_stencil_nodes": bed.nodes.numpy(), "reference_stencil_weights": bed.weights.numpy(),
            "reference_stencil_gradients": bed.gradients.numpy(),
            "surface_vertices_world": vertices, "surface_triangles": triangles,
            "surface_triangle_component_ids": face_components,
            "surface_component_provenance": metadata["partition_rule"], "surface_timestamp_s": self.time_s})
        return {**result, "success": result["status"] == "PASS_TASK_METRICS",
                "episode_complete": self.time_s >= self.task.duration_s-1e-9, "runtime_source": self.runtime_source}

    def close(self):
        if not self._closed:
            self.backend.close()
            self._closed = True
            self._history.clear()
            self._observation = None


@contextmanager
def open_cutting_env(task_id="CUT-01", *, runtime="isaac_sim", output_dir, seed=42,
                     action_mode=ACTION_MODES[0], observation_profile="vision_proprio",
                     include_global_camera=False, camera_layout="wrist_d405_global_d455", **backend_options):
    """Start a pinned live runtime in its dedicated venv, then close it on exit.

    Use a fresh output directory. Isaac Sim defaults to real native SingleView
    RGB-D; the separate Lab kit-less route explicitly uses ideal Warp RGB-D.
    A process should own only one Isaac runtime at a time.
    """
    from .recording import manifest, new_output, write_json
    if runtime not in ("isaac_sim", "isaac_lab") or action_mode not in ACTION_MODES:
        raise ValueError("Choose a supported live runtime and action mode")
    make_task(task_id, seed=seed)
    output = new_output(output_dir)
    wrapper = None
    if runtime == "isaac_sim":
        from .isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app
        mode = "joint" if action_mode == ACTION_MODES[1] else "cartesian"
        config = IsaacSimCuttingConfig(task_id=task_id, seed=seed, action_mode=mode,
            **{"camera_mode": "native_single_view", "camera_layout": camera_layout, **backend_options})
        write_json(output/"manifest.json", manifest({**asdict(config),
            "formal_observation": {"profile": observation_profile, "include_global_camera": bool(include_global_camera)}}, "live_isaac_sim_shared_api"))
        app = launch_app(config, output)
        try:
            wrapper = CuttingTaskEnv(LiveIsaacSimCutting(app, config, output), task_id, seed=seed,
                action_mode=action_mode, observation_profile=observation_profile,
                include_global_camera=include_global_camera)
            yield wrapper
        finally:
            if wrapper:
                wrapper.close()
            else:
                app.close(wait_for_replicator=False)
    elif runtime == "isaac_lab":
        from argparse import Namespace
        from isaaclab.app import launch_simulation
        from .isaac_lab_env import IsaacLabCuttingEnv, LabCuttingConfig, make_lab_cfg
        from .workcell import export
        static = output/"workcell"
        static.mkdir()
        export(Namespace(variant="D0", family="dome", seed=seed, spacing=.002, asset_root=None), static)
        config = LabCuttingConfig(task_id=task_id, seed=seed, action_mode=action_mode,
            static_scene=str((static/"workcell.usda").resolve()), camera_layout=camera_layout,
            include_global_camera=include_global_camera, **backend_options)
        write_json(output/"manifest.json", manifest({**asdict(config),
            "formal_observation": {"profile": observation_profile, "include_global_camera": bool(include_global_camera)}}, "live_isaac_lab_shared_api"))
        cfg = make_lab_cfg(config)
        with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": config.device}):
            wrapper = CuttingTaskEnv(IsaacLabCuttingEnv(cfg, config), task_id, seed=seed,
                action_mode=action_mode, observation_profile=observation_profile,
                include_global_camera=include_global_camera)
            try:
                yield wrapper
            finally:
                wrapper.close()
    else:
        raise ValueError("runtime must be isaac_sim or isaac_lab")

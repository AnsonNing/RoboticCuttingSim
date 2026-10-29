"""Tool-call style control for LLM agents, on top of CuttingTaskEnv.

Every method returns a JSON-serializable dict, so an agent framework can expose
them directly as tools. Motions are executed as the same bounded
``cartesian_edge_delta`` actions a policy or RL agent emits (WaypointPolicy),
one control step at a time, so per-step limits, IK rejection and termination
apply unchanged. ``describe_task`` gives the a-priori task description
(goal and nominal workpiece geometry); it never exposes oracle material state.
"""
from pathlib import Path
import struct
import zlib

import numpy as np
from scipy.spatial.transform import Rotation

from .control import DELTA_MODE, ActionLimits
from .policies import Waypoint, WaypointPolicy, edge_from_observation


def write_png(path, rgb):
    """Minimal dependency-free RGB8 PNG writer (agents read images from files)."""
    image = np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image")
    height, width = image.shape[:2]
    raw = b"".join(b"\x00"+image[row].tobytes() for row in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data))+kind+data+struct.pack(">I", zlib.crc32(kind+data) & 0xffffffff)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6))+chunk(b"IEND", b""))
    return str(path)


def _floats(values, digits=5):
    return [round(float(v), digits) for v in np.ravel(values)]


class CuttingAgentAPI:
    def __init__(self, env):
        if env.action_mode != DELTA_MODE and env.action_limits is None:
            env.action_limits = ActionLimits()
        self.env = env
        self.limits = env.action_limits or ActionLimits()
        self.observation, self.last_info, self.done = None, {}, False

    # ----- information -------------------------------------------------
    def describe_task(self):
        task = self.env.task
        T = task.T_world_workpiece
        goals = {"sever": "Cut the workpiece into two pieces at its target plane.",
                 "flatten": "Remove the material above the target plane (cut off the bump) without cutting below it."}
        return {"task_id": task.task_id, "goal": goals[task.goal_kind],
                "workpiece_center_world_m": _floats(T[:3, 3]),
                "workpiece_size_m": _floats([task.geometry.length_m, task.geometry.width_m, task.geometry.base_height_m]),
                "bumps": [{"center_local_m": _floats([b.x_m, b.y_m]), "height_m": round(b.height_m, 5),
                           "radius_m": _floats([b.radius_x_m, b.radius_y_m])} for b in task.geometry.bumps],
                "target_plane_world": _floats(task.target_plane_world()),
                "edge_frame": "x = cutting (advance) direction, y = along the edge, z = blade normal",
                "episode_s": task.duration_s, "control_dt_s": task.control_dt_s,
                "step_limits": {"translation_m": self.limits.max_translation_m,
                                "rotation_deg": round(float(np.degrees(self.limits.max_rotation_rad)), 3)},
                "tool": task.tool_profile.to_dict()["profile_id"]}

    def status(self):
        edge = edge_from_observation(self.observation, self.env.task) if self.observation else None
        info = self.last_info
        return {"time_s": round(self.env.time_s, 4), "remaining_s": round(self.env.task.duration_s-self.env.time_s, 4),
                "done": self.done,
                "edge_position_m": None if edge is None else _floats(edge[:3]),
                "edge_rpy_deg": None if edge is None else _floats(Rotation.from_quat(edge[3:]).as_euler("xyz", degrees=True), 3),
                "termination_reason": info.get("termination_reason"), "ik_status": info.get("ik_status"),
                "consecutive_ik_failures": info.get("consecutive_ik_failures", 0)}

    def observe(self):
        """Measured state (and images when the env provides them)."""
        result = self.status()
        if self.observation is not None and "rgb" in self.observation:
            result["cameras"] = sorted(self.observation["rgb"])
        return result

    def save_views(self, directory, prefix=None):
        """Write the current RGB camera views to PNG files and return their paths.

        Depth stays in the observation; this is for agents that look at images.
        Backends without cameras (kinematic) return an empty list and a reason.
        """
        if self.observation is None or "rgb" not in self.observation:
            return {"images": [], "reason": "This environment provides no camera images (use vision_proprio)"}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = prefix or f"t{self.env.time_s:07.3f}"
        paths = [write_png(directory/f"{stem}_{name}.png", rgb) for name, rgb in sorted(self.observation["rgb"].items())]
        return {"images": paths, "time_s": round(self.env.time_s, 4)}

    # ----- control -----------------------------------------------------
    def reset(self, seed=None):
        self.observation, info = self.env.reset(seed=seed)
        self.last_info, self.done = info, False
        return {**self.status(), "runtime_source": info.get("runtime_source")}

    def _run(self, waypoints, max_steps):
        policy = WaypointPolicy(self.env.task, list(waypoints), self.limits)
        steps, rejected, total_reward = 0, 0, 0.
        while not policy.done and not self.done and steps < max_steps:
            action = policy(self.observation)
            if policy.done:
                break
            self.observation, reward, terminated, truncated, info = self.env.step(action)
            self.last_info, steps = info, steps+1
            total_reward += reward
            rejected += info.get("ik_status") == "rejected"
            self.done = bool(terminated or truncated)
        return {"steps": steps, "reached": policy.done, "ik_rejected_steps": int(rejected),
                "reward": round(total_reward, 6), **self.status()}

    def _edge(self):
        return edge_from_observation(self.observation, self.env.task)

    def move_edge_to(self, position_m, rpy_deg=None, *, cut=False, speed_m_s=None, max_steps=200, cut_index=0):
        """Move the edge center to a WORLD position (and orientation, XYZ Euler deg)."""
        edge = self._edge()
        rotation = Rotation.from_quat(edge[3:]) if rpy_deg is None else Rotation.from_euler("xyz", rpy_deg, degrees=True)
        pose = tuple(np.r_[np.asarray(position_m, float), rotation.as_quat()])
        return self._run([Waypoint(pose, cut=bool(cut), speed_m_s=speed_m_s, cut_index=int(cut_index))], max_steps)

    def move_edge_by(self, delta_m=(0, 0, 0), delta_rpy_deg=(0, 0, 0), *, frame="world", cut=False,
                     speed_m_s=None, max_steps=200, cut_index=0):
        """Relative move: translation in the world or edge frame, rotation about the edge center."""
        edge = self._edge()
        rotation = Rotation.from_quat(edge[3:])
        step = Rotation.from_euler("xyz", delta_rpy_deg, degrees=True)
        delta = np.asarray(delta_m, float)
        if frame == "edge":
            position, final = edge[:3]+rotation.apply(delta), rotation*step
        elif frame == "world":
            position, final = edge[:3]+delta, step*rotation
        else:
            raise ValueError("frame must be 'world' or 'edge'")
        return self._run([Waypoint(tuple(np.r_[position, final.as_quat()]), cut=bool(cut), speed_m_s=speed_m_s,
                                   cut_index=int(cut_index))], max_steps)

    def cut_along(self, direction="advance", distance_m=.03, *, speed_m_s=.01, saw_amplitude_m=0.,
                  saw_period_s=.6, max_steps=400, cut_index=0):
        """Cut: move the armed edge a distance along 'advance' (edge x) or a WORLD vector,
        optionally sawing back and forth along the edge line."""
        edge = self._edge()
        rotation = Rotation.from_quat(edge[3:])
        vector = rotation.apply([1., 0., 0.]) if isinstance(direction, str) and direction == "advance" \
            else np.asarray(direction, float)
        vector = vector/np.linalg.norm(vector)
        end = tuple(np.r_[edge[:3]+vector*float(distance_m), edge[3:]])
        return self._run([Waypoint(end, cut=True, speed_m_s=speed_m_s, saw_amplitude_m=saw_amplitude_m,
                                   saw_period_s=saw_period_s, cut_index=int(cut_index))], max_steps)

    def wait(self, steps=10):
        """Hold the current pose (cutting disarmed) for some control steps."""
        return self._hold(int(steps))

    def _hold(self, steps):
        count = 0
        while count < steps and not self.done:
            self.observation, _, terminated, truncated, self.last_info = self.env.step(
                {"mode": DELTA_MODE, "frame": "world", "target": np.zeros(6), "cut_enabled": False})
            count += 1
            self.done = bool(terminated or truncated)
        return {"steps": count, **self.status()}

    def finish_episode(self):
        """Hold still (cutting disarmed) until the episode time limit."""
        return self._hold(int(round((self.env.task.duration_s-self.env.time_s)/self.env.task.control_dt_s)))

    def set_gripper(self, command):
        options = getattr(self.env.backend, "options", getattr(self.env.backend, "config", None))
        if getattr(options, "tool_attachment_mode", None) != "physical_grasp":
            return {"ok": False, "reason": "This runtime has no physical gripper"}
        self.observation, _, terminated, truncated, self.last_info = self.env.step({"gripper": command})
        self.done = bool(terminated or truncated)
        return {"ok": True, **self.status()}

    def evaluate(self):
        result = self.env.evaluate()
        keep = ("status", "success", "episode_complete", "checks", "runtime_source")
        summary = {key: result.get(key) for key in keep if key in result}
        metrics = result.get("metrics", {})
        for name in ("final_components", "substantial_parts", "two_main_parts_mass_fraction",
                     "removed_reference_above_plane_mass_fraction", "maximum_reference_midpoint_plane_distance_m"):
            if name in metrics:
                summary[name] = metrics[name]
        return summary

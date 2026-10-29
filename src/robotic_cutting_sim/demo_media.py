"""Pure helpers for rendering recorded demo episodes into videos.

Read-only consumers of an episode directory written by
``scripts/run_wp10_live.py``. Nothing here changes a recording, a physics result
or an evaluation; component colouring is an oracle debug view and is never an
observation.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from .cut_topology import mechanical_components

ORACLE_LABEL = "oracle debug view (not an observation)"

# Dark-surface categorical steps (validated reference palette, first three slots
# validate all-pairs); minor fragments are a neutral grey, never a fourth hue.
SURFACE = "#1a1a19"
TEXT_PRIMARY = "#f0efec"
TEXT_SECONDARY = "#c3c2b7"
COMPONENT_COLORS = ("#3987e5", "#d95926", "#199e70")
MINOR_COLOR = "#8a8984"

FORCE_DEFINITION = ("|F| = ||current_material_reaction_impulse_world[:3]|| / control_dt: mean material "
                    "reaction force on the blade over one control step (finger-box reactions excluded)")

RUNTIME_NAMES = {
    "live_isaac_sim": "Live Isaac Sim Native",
    "live_isaac_lab": "Live Isaac Lab",
    "kinematic_ideal_tracking": "Kinematic ideal tracking (not live)",
    "measured_trajectory_material_replay": "Measured-trajectory material replay (not live)",
}

_STREAM = re.compile(r"^(?P<name>.+)_(?P<index>\d{4})$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_display_name(runtime_source):
    if not runtime_source:
        return "unknown runtime"
    return RUNTIME_NAMES.get(runtime_source, str(runtime_source))


def discover_streams(observation_names):
    """Map camera stream name -> sorted frame indices from ``<name>_<iiii>.npz`` stems."""
    streams = {}
    for stem in observation_names:
        match = _STREAM.match(Path(stem).stem)
        if match:
            streams.setdefault(match["name"], []).append(int(match["index"]))
    return {name: sorted(indices) for name, indices in sorted(streams.items())}


CLOSEUP_TITLE = "Close-up (presentation camera, not an observation)"
LAYOUT_DEFAULT = "large=overview or Global D455; secondary=Global D455 when an overview exists, else Wrist D405_1"
LAYOUT_CLOSEUP = "large=overview or Global D455; secondary=presentation close-up (closeup_*)"


def choose_panels(streams):
    """Panel -> stream assignment. The large panel is an overview camera when one
    was recorded, else the Global D455. The secondary RGB slot shows a recorded
    presentation close-up (``closeup_*``) when one exists; otherwise the Global
    D455 when it is not already the large panel, else the other wrist camera, so
    no stream is duplicated. The ``closeup`` key is present only when a close-up
    stream was recorded, so older episodes keep their exact assignment."""
    overview = next((name for name in streams if name.startswith("overview")), None)
    closeup = next((name for name in streams if name.startswith("closeup")), None)
    large = overview or ("global_d455" if "global_d455" in streams else None)
    secondary = "global_d455" if overview and "global_d455" in streams else (
        "wrist_d405_1" if "wrist_d405_1" in streams else None)
    wrist = "wrist_d405_0" if "wrist_d405_0" in streams else None
    panels = dict(large=large, secondary=closeup or secondary, wrist_rgb=wrist, wrist_depth=wrist, overview=overview)
    if closeup:
        panels["closeup"] = closeup
    return panels


def panel_layout(panels):
    """Receipt description of the layout rule that produced ``panels``."""
    return LAYOUT_CLOSEUP if panels.get("closeup") else LAYOUT_DEFAULT


def panel_title(stream, names):
    """Panel title for a secondary stream; a close-up is always marked presentation-only."""
    if stream and stream.startswith("closeup"):
        return CLOSEUP_TITLE
    return names.get(stream, stream or "secondary RGB")


def cut_count(codes):
    """Cut digits needed by the topology audit: 2 if any code >= 3, else 1."""
    return 2 if int(np.max(codes)) >= 3 else 1


class ComponentCache:
    """mechanical_components() memoised per unique code vector."""

    def __init__(self, nodes, weights, gradients=None, cuts=1):
        self.nodes, self.weights, self.gradients, self.cuts = nodes, weights, gradients, cuts
        self.cache, self.calls = {}, 0

    def __call__(self, codes):
        codes = np.ascontiguousarray(codes)
        key = hashlib.sha1(codes.tobytes()).hexdigest()
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = mechanical_components(self.nodes, self.weights, codes, self.cuts,
                                                    gradients=self.gradients)
        return self.cache[key]


def component_order(labels, reference_positions, minor_fraction=.01, slots=len(COMPONENT_COLORS)):
    """Stable colour slots for components.

    Components of at least ``minor_fraction`` of the particles are ordered by the
    reference-configuration centroid along the workpiece's longest axis (so a
    piece keeps its colour as another splits off); at most ``slots`` get a hue and
    every other component is a minor fragment (slot -1).
    Returns (slot per component id, sizes).
    """
    labels = np.asarray(labels)
    reference = np.asarray(reference_positions, float)
    count = int(labels.max()) + 1 if labels.size else 0
    sizes = np.bincount(labels, minlength=count)
    axis = int(np.argmax(np.ptp(reference, axis=0))) if len(reference) else 0
    centroid = np.bincount(labels, reference[:, axis], minlength=count) / np.maximum(sizes, 1)
    major = [c for c in range(count) if sizes[c] >= minor_fraction * len(labels)]
    major = sorted(sorted(major, key=lambda c: -sizes[c])[:slots], key=lambda c: centroid[c])
    slot = np.full(count, -1, int)
    slot[major] = np.arange(len(major))
    return slot, sizes


def hex_rgba(color, alpha=1.):
    color = color.lstrip("#")
    return np.array([int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha])


def component_colors(labels, reference_positions, minor_fraction=.01):
    """Per-particle RGBA from component labels plus the slot of each component."""
    slot, sizes = component_order(labels, reference_positions, minor_fraction)
    palette = np.array([hex_rgba(c) for c in COMPONENT_COLORS] + [hex_rgba(MINOR_COLOR)])
    return palette[slot[np.asarray(labels)]], slot, sizes


def edge_segment(edge_pose_xyz_xyzw, half_length):
    """World endpoints of the measured cutting edge (edge-frame local +y axis)."""
    pose = np.asarray(edge_pose_xyz_xyzw, float)
    axis = Rotation.from_quat(pose[3:7]).apply([0., 1., 0.])
    return np.stack([pose[:3] - half_length * axis, pose[:3] + half_length * axis])


def edge_half_length(task):
    profile = (task or {}).get("tool_profile") or {}
    for value in (profile.get("half_width_m"), (task or {}).get("blade_half_width_m"),
                  None if profile.get("edge_length_m") is None else profile["edge_length_m"] / 2):
        if value is not None:
            return float(value)
    return None


def reaction_force_n(impulse_world, control_dt):
    impulse = np.asarray(impulse_world, float)
    return float(np.linalg.norm(impulse[:3]) / control_dt)


def measured_rate(timestamps):
    """Capture rate measured from actual frame timestamps."""
    t = np.asarray([v for v in timestamps if v is not None and np.isfinite(v)], float)
    if len(t) < 2:
        return dict(frames=int(len(t)), rate_hz=None, interval_s=None)
    dt = np.diff(t)
    return dict(frames=int(len(t)), rate_hz=float(1 / np.median(dt)),
                interval_s=dict(median=float(np.median(dt)), min=float(dt.min()), max=float(dt.max())),
                span_s=float(t[-1] - t[0]), monotonic=bool((dt > 0).all()))


def status_line(label=None, result=None):
    """Final evaluation status text; an explicit label wins over result.json."""
    if label:
        return str(label), "--label"
    result = result or {}
    task = result.get("task_status")
    execution = result.get("status")
    if task is None and execution is None:
        return "no evaluation recorded", "missing"
    parts = [f"task {task}" if task else None, f"execution {execution}" if execution else None]
    return ", ".join(p for p in parts if p), "result.json"


def frame_annotation(*, runtime, controller, time_s, active_cut, cut_index, status, task_id=None):
    """Header text lines drawn on every frame."""
    cut = f"CUT ON (cut_index {cut_index})" if active_cut else f"CUT OFF (cut_index {cut_index})"
    first = " | ".join(p for p in (task_id, runtime, f"controller: {controller}") if p)
    second = f"sim t = {time_s:6.2f} s | {cut} | final eval: {status}"
    return [first, second]


def nearest_rows(row_times, frame_times):
    """Index of the last action row at or before each frame time (-1 = none)."""
    row_times = np.asarray(row_times, float)
    return np.searchsorted(row_times, np.asarray(frame_times, float) + 1e-9, side="right") - 1


def gif_plans(frames, fps, width):
    """Successively cheaper (stride, fps, width) GIF settings to meet a size budget."""
    plans = []
    for stride, w in ((1, width), (1, int(width * .8)), (2, int(width * .8)), (2, int(width * .66)),
                      (3, int(width * .5))):
        if stride == 1 or frames // stride >= 4:
            plans.append(dict(stride=stride, fps=max(fps / stride, 1.), width=w - w % 2))
    return plans


def render_receipt(*, episode, out, inputs, frames, rate, video_fps, gif, panels, components,
                   cuts, particle_stride, status, status_source, git, wall_s, command, outputs,
                   label_source_runtime, versions):
    return dict(
        schema="demo_render_receipt_v1",
        episode_dir=str(episode), output_dir=str(out), command=list(command),
        inputs=inputs, frame_count=int(frames), measured_capture_rate=rate,
        video_fps=float(video_fps), gif=gif, panels=panels,
        particle_view=dict(label=ORACLE_LABEL, stride=int(particle_stride), cuts=int(cuts),
                           component_function="robotic_cutting_sim.cut_topology.mechanical_components "
                                              "(nodes, weights, codes, cuts, gradients=...)",
                           component_count_per_frame=[int(c) for c in components]),
        force_definition=FORCE_DEFINITION, status_line=status, status_source=status_source,
        runtime_label=label_source_runtime, git=git, wall_time_s=float(wall_s), outputs=outputs,
        versions=versions, recording_modified=False)


def load_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]

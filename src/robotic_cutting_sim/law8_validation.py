"""WP-11 Phase F helpers: law-8 boundary-case validation (``configs/wp11/phaseF_protocol_v1.json``).

Read-only, oracle-side tools used by ``scripts/run_phase_f_material.py``:

* the Phase F task variants (``configs/cutting_tasks_v4_phaseF.json``; opt-in, never
  registered with the live runtimes) and the F2.11 configured scene/goal;
* ``ScheduledPolicy``: a time-scheduled kinematic edge motion (rotation about the
  edge centre, rocking, draw in place) emitting the same bounded
  ``cartesian_edge_delta`` actions as every other controller; a schedule that would
  exceed the action limits is rejected, never clamped silently;
* geometric/topological diagnostics: the in-plane region the moving edge actually
  swept, lattice bonds severed by conflicting codes, code-0 stencil bridges, mass
  conservation against the task sampler, contact-window forces, plowing fractions,
  cross-component proximity and digit persistence.

None of this changes a cutting law, a parameter or a threshold. Material state is
evaluation-only, never an observation.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .control import DELTA_MODE
from .cut_topology import conflicting, conflicting_packed, decode_digits
from .tasks import (PROTOCOL_PATH, V4_TASK_IDS, Bump, FixtureBox, HeightFieldSpec, MaterialParameters,
                    ProfileTaskSpec, ProfileTaskSpecV4, make_task, rake_edge_orientation_xyzw)

PHASE_F_TASKS_PATH = Path(__file__).resolve().parents[2]/"configs"/"cutting_tasks_v4_phaseF.json"
PHASE_F_PROTOCOL_PATH = Path(__file__).resolve().parents[2]/"configs"/"wp11"/"phaseF_protocol_v1.json"


# ----------------------------------------------------------------------------- tasks

def phase_f_tasks():
    return json.loads(PHASE_F_TASKS_PATH.read_text())


def _v1():
    return json.loads(PROTOCOL_PATH.read_text())


def phase_f_task_config(task_id):
    """v1 base task entry with the Phase F overrides (same merge rule as ``tasks._v4_task_config``)."""
    entry = phase_f_tasks()["tasks"][task_id]
    config = deepcopy(_v1()[entry["base"]])
    config.update({key: deepcopy(value) for key, value in entry.items() if key not in ("base", "tool_profile")})
    if "blade_rake_deg" in entry:
        if "edge_orientation_local_xyzw" in entry:
            raise ValueError("A Phase F task declares either a rake or an explicit edge orientation, not both")
        config["edge_orientation_local_xyzw"] = list(rake_edge_orientation_xyzw(
            entry["blade_rake_deg"], entry.get("edge_tilt_deg", 0.), entry.get("edge_skew_deg", 0.)))
    return entry["base"], entry["tool_profile"], config


@dataclass(frozen=True)
class ProfileTaskSpecV4F(ProfileTaskSpecV4):
    """A Phase F task variant (``cutting_tasks_v4_phaseF.json``); v1 base thresholds unchanged."""

    TASK_IDS: ClassVar[tuple] = ("TRIM-02-K2-R14", "TRIM-02-K2-R16", "CUT-04-K2-W60")

    @property
    def protocol_path(self):
        return PHASE_F_TASKS_PATH

    def protocol_document(self):
        return {"protocol_id": phase_f_tasks()["protocol_id"], "common": _v1()["common"],
                self.task_id: phase_f_task_config(self.task_id)[2]}

    def to_dict(self):
        entry = phase_f_tasks()["tasks"][self.task_id]
        declared = None if self.blade_rake_deg is None else {
            "blade_rake_deg": self.blade_rake_deg, "edge_tilt_deg": entry.get("edge_tilt_deg", 0.),
            "edge_skew_deg": entry.get("edge_skew_deg", 0.), "target_normal_local": list(self.target_normal_local),
            "cutting_direction_local": [1., 0., 0.], "convention": "docs/adr/0013 (rho > 0 raises the blade body behind the edge)"}
        return {**ProfileTaskSpec.to_dict(self), "derived_from": entry["derived_from"],
                "declared_orientation_metrics": declared, "phase_f_variant": True}


def make_phase_f_task(task_id, seed=42, *, particle_spacing_m=None, grid_dx_m=None):
    """A cutting_tasks_v4 task or a Phase F variant; ``grid_dx_m`` is an explicit numerical override (F3)."""
    from dataclasses import replace
    from .tool_profile import PROFILES
    if task_id in V4_TASK_IDS:
        task = make_task(task_id, seed, particle_spacing_m=particle_spacing_m)
    elif task_id in ProfileTaskSpecV4F.TASK_IDS:
        v1 = _v1()
        base, profile_id, config = phase_f_task_config(task_id)
        profile = PROFILES[profile_id]
        geometry, episode = config["geometry"], v1["episode"]
        task = ProfileTaskSpecV4F(task_id, int(seed), "fixed",
            HeightFieldSpec(geometry["length_m"], geometry["width_m"], geometry["base_height_m"],
                            tuple(Bump(**row) for row in geometry["bumps"])),
            config["particle_spacing_m"] if particle_spacing_m is None else float(particle_spacing_m), config["grid_dx_m"],
            tuple(FixtureBox(row["name"], tuple(row["center_m"]), tuple(row["half_extents_m"]))
                  for row in config["fixture_boxes_local"]),
            (0, 0, 0, 0, 0, 0, 1), v1["common"]["table_top_world_z_m"], tuple(profile.hand_from_edge()),
            episode["duration_s"], episode["control_dt_s"], MaterialParameters(**v1["material"]),
            config["goal_kind"], tuple(config["target_origin_local_m"]), tuple(config["target_normal_local"]),
            config.get("edge_clearance_m", 0.), profile.blade_depth_m, profile.thickness_m, profile.half_width_m,
            tuple(config["edge_orientation_local_xyzw"]), tuple(tuple(row) for row in config["edge_position_knots_local_txyz"]),
            tuple(episode["active_cut_interval_s"]), tool_profile_id=profile.profile_id, base_task_id=base,
            blade_rake_deg=config.get("blade_rake_deg"))
    else:
        raise ValueError(f"Unknown Phase F task id {task_id!r}")
    if grid_dx_m is not None:
        task = replace(task, grid_dx_m=float(grid_dx_m))
    return task


def configured_phase_f_scene_goal(name):
    """(scene dict, goal dict) of a Phase F configured task: the base scene with the declared overrides."""
    from .scene_config import load_json
    entry = phase_f_tasks()["configured_tasks"][name]
    scene = deepcopy(load_json(entry["scene_base"]))
    for key, value in entry["scene_overrides"].items():
        scene[key] = deepcopy(value)
    return scene, deepcopy(entry["goal"])


# ----------------------------------------------------------------------------- scheduled motion

@dataclass(frozen=True)
class Segment:
    """One timed piece of a schedule: ``pose(s)`` (WORLD edge XYZ+XYZW) for s in [0, duration_s]."""
    label: str
    duration_s: float
    pose: Callable
    cut: bool = False
    cut_index: int = 0


def _pose(position, rotation):
    return np.r_[np.asarray(position, float), rotation.as_quat()]


def hold(label, position, rotation, duration_s, *, cut=False):
    position = np.asarray(position, float)
    return Segment(label, float(duration_s), lambda s: _pose(position, rotation), cut)


def line(label, start, end, rotation, speed_m_s, dt, *, cut=False):
    """Straight move at constant speed; the duration is rounded up to whole control steps."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    distance = float(np.linalg.norm(end-start))
    steps = max(1, int(np.ceil(distance/(speed_m_s*dt)-1e-9)))
    duration = steps*dt
    return Segment(label, duration, lambda s: _pose(start+(end-start)*min(max(s/duration, 0.), 1.), rotation), cut)


def triangle_wave(s, amplitude, period):
    """0 -> +A -> -A -> 0 over one period (piecewise linear, constant |rate| = 4A/period)."""
    phase = (s/period) % 1.
    if phase < .25:
        return amplitude*4*phase
    if phase < .75:
        return amplitude*(2-4*phase)
    return amplitude*(4*phase-4)


class ScheduledPolicy:
    """Bounded delta actions that follow a time schedule of ``Segment``s exactly (ideal tracking).

    At observation time t the target is the scheduled pose at t + control dt; the
    action is the WORLD delta from the measured edge to it. A delta beyond the
    action limits raises (the schedule is invalid) instead of being clamped.
    ``phase`` is the label of the segment the latest target belongs to.
    """

    def __init__(self, task, segments, limits, *, edge_estimator=None):
        self.task, self.segments, self.limits = task, list(segments), limits
        self.dt = task.control_dt_s
        self.starts = np.cumsum([0.]+[s.duration_s for s in self.segments])
        self.edge_estimator = edge_estimator
        self.phase, self.done = self.segments[0].label, False
        self.max_translation_m, self.max_rotation_rad = 0., 0.

    @property
    def duration_s(self):
        return float(self.starts[-1])

    def target(self, time_s):
        """(pose, segment or None) scheduled at ``time_s`` (segment k owns (start_k, end_k])."""
        if time_s > self.starts[-1]+1e-9:
            last = self.segments[-1]
            return last.pose(last.duration_s), None
        k = int(np.clip(np.searchsorted(self.starts, time_s-1e-9, side="left")-1, 0, len(self.segments)-1))
        segment = self.segments[k]
        return segment.pose(min(max(time_s-self.starts[k], 0.), segment.duration_s)), segment

    def __call__(self, observation):
        from .policies import edge_from_observation
        edge = (edge_from_observation(observation, self.task) if self.edge_estimator is None
                else np.asarray(self.edge_estimator(observation), float))
        time_s = float(observation["timestamp_s"])
        pose, segment = self.target(time_s+self.dt)
        self.done = segment is None
        self.phase = "done" if segment is None else segment.label
        translation = pose[:3]-edge[:3]
        rotation = (Rotation.from_quat(pose[3:])*Rotation.from_quat(edge[3:]).inv()).as_rotvec()
        moved, turned = float(np.linalg.norm(translation)), float(np.linalg.norm(rotation))
        if moved > self.limits.max_translation_m+1e-12 or turned > self.limits.max_rotation_rad+1e-12:
            raise ValueError(f"Schedule exceeds the action limits at t = {time_s:.3f} s ({self.phase}): "
                             f"{moved*1e3:.4f} mm, {np.rad2deg(turned):.4f} deg per step")
        self.max_translation_m = max(self.max_translation_m, moved)
        self.max_rotation_rad = max(self.max_rotation_rad, turned)
        cut = segment is not None and segment.cut
        return {"mode": DELTA_MODE, "frame": "world", "target": np.r_[translation, rotation], "cut_enabled": bool(cut),
                **({"cut_index": segment.cut_index} if cut and segment.cut_index else {})}

    def describe(self):
        return [dict(label=s.label, start_s=float(t), duration_s=s.duration_s, cut=s.cut, cut_index=s.cut_index,
                     start_pose=s.pose(0.).tolist(), end_pose=s.pose(s.duration_s).tolist())
                for s, t in zip(self.segments, self.starts)]


# ----------------------------------------------------------------------------- swept region

def edge_sweep_points(poses, half_length_m, *, cell_m=2.5e-4):
    """Points on the edge line for consecutive WORLD edge poses (T, 7), interpolated in time.

    Between consecutive poses the centre is interpolated linearly and the rotation
    by slerp, with enough sub-samples that no edge point moves more than half a
    cell; along the edge the points are at most one cell apart.
    """
    poses = np.asarray(poses, float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses):
        raise ValueError("Need (T, 7) edge poses")
    ls = np.linspace(-half_length_m, half_length_m, int(np.ceil(2*half_length_m/cell_m))+1)
    rotations = Rotation.from_quat(poses[:, 3:])
    chunks = [poses[:1, :3]+np.outer(ls, rotations[0].apply([0., 1., 0.]))]
    for i in range(len(poses)-1):
        a, b = poses[i], poses[i+1]
        angle = (rotations[i+1]*rotations[i].inv()).magnitude()
        reach = float(np.linalg.norm(b[:3]-a[:3]))+angle*half_length_m
        n = max(1, int(np.ceil(reach/(.5*cell_m))))
        fractions = np.linspace(0., 1., n+1)[1:]
        slerp = Slerp([0., 1.], Rotation.concatenate([rotations[i], rotations[i+1]]))
        lines = slerp(fractions).apply([0., 1., 0.])
        centres = a[:3]+np.outer(fractions, b[:3]-a[:3])
        chunks.append((centres[:, None, :]+ls[None, :, None]*lines[:, None, :]).reshape(-1, 3))
    return np.concatenate(chunks)


@dataclass
class SweptRegion:
    """Rasterized in-plane region (blade-plane coordinates u along the edge, w along the advance)."""
    origin: np.ndarray
    u: np.ndarray
    w: np.ndarray
    cell_m: float
    low: np.ndarray
    occupied: np.ndarray
    distance_cells: np.ndarray
    out_of_plane_max_m: float

    def coordinates(self, points):
        p = np.asarray(points, float)-self.origin
        return np.c_[p@self.u, p@self.w]

    def distance_m(self, points):
        """In-plane distance (m) from each point's projection to the swept region (0 inside)."""
        index = np.floor((self.coordinates(points)-self.low)/self.cell_m).astype(int)
        inside = ((index >= 0) & (index < np.asarray(self.occupied.shape))).all(1)
        out = np.full(len(index), np.inf)
        out[inside] = self.distance_cells[index[inside, 0], index[inside, 1]]*self.cell_m
        return out

    def extent(self):
        cells = np.argwhere(self.occupied)
        lo, hi = self.low+cells.min(0)*self.cell_m, self.low+(cells.max(0)+1)*self.cell_m
        return {"u_m": [float(lo[0]), float(hi[0])], "w_m": [float(lo[1]), float(hi[1])],
                "area_m2": float(self.occupied.sum()*self.cell_m**2)}


def swept_region(pose_chains, half_length_m, *, origin, edge_line, advance, cell_m=2.5e-4, pad_m=.08):
    """Region of the blade plane covered by the edge over chains of consecutive poses (rasterized, with an EDT).

    ``pose_chains`` is a list of (T, 7) arrays; poses are interpolated within a chain only.
    """
    from scipy.ndimage import distance_transform_edt
    u, w = np.asarray(edge_line, float), np.asarray(advance, float)
    u, w = u/np.linalg.norm(u), w/np.linalg.norm(w)
    if abs(u@w) > 1e-6:
        raise ValueError("Edge line and advance must be orthogonal")
    origin = np.asarray(origin, float)
    points = np.concatenate([edge_sweep_points(chain, half_length_m, cell_m=cell_m) for chain in pose_chains])
    p = points-origin
    plane = np.c_[p@u, p@w]
    normal = np.cross(u, w)
    low = plane.min(0)-pad_m
    shape = np.ceil((plane.max(0)+pad_m-low)/cell_m).astype(int)+1
    occupied = np.zeros(shape, bool)
    index = np.floor((plane-low)/cell_m).astype(int)
    occupied[index[:, 0], index[:, 1]] = True
    distance = distance_transform_edt(~occupied)
    return SweptRegion(origin, u, w, float(cell_m), low, occupied, distance, float(np.abs(p@normal).max()))


def label_region_report(region, positions, *, tolerance_m, what):
    """Distances of labelled particles (given positions) outside the swept region, against a tolerance."""
    if not len(positions):
        return {"what": what, "particles": 0, "outside_tolerance": 0, "maximum_outside_m": None, "tolerance_m": tolerance_m}
    distance = region.distance_m(positions)
    finite = np.isfinite(distance)
    worst = float(distance.max()) if finite.all() else float("inf")
    return {"what": what, "particles": int(len(distance)), "outside_region": int(np.count_nonzero(distance > 0)),
            "outside_tolerance": int(np.count_nonzero(~finite | (distance > tolerance_m))),
            "maximum_outside_m": worst, "percentiles_outside_m": {str(q): float(np.percentile(distance[finite], q))
                                                                  for q in (50, 90, 99, 100)} if finite.any() else None,
            "tolerance_m": float(tolerance_m), "cell_m": region.cell_m,
            "scope": "In-plane distance of the projection onto the blade plane to the rasterized region the edge "
                     "line swept while armed (out-of-plane |s| is the law's band, not part of this statistic)"}


# ----------------------------------------------------------------------------- bonds, bridges, mass

def lattice_bonds(lattice):
    """(a, b) particle indices of every 6-neighbour reference lattice bond (vectorized)."""
    lattice = np.asarray(lattice, np.int64)
    lattice = lattice-lattice.min(0)
    dims = lattice.max(0)+2
    key = lambda v: (v[:, 0]*dims[1]+v[:, 1])*dims[2]+v[:, 2]  # noqa: E731
    keys = key(lattice)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    a, b = [], []
    for step in np.eye(3, dtype=np.int64):
        target = key(lattice+step)
        position = np.searchsorted(sorted_keys, target)
        position = np.minimum(position, len(sorted_keys)-1)
        hit = sorted_keys[position] == target
        a.append(np.flatnonzero(hit))
        b.append(order[position[hit]])
    return np.concatenate(a), np.concatenate(b)


def severed_bonds(lattice, codes, cuts=1, *, packed=False):
    """(a, b) of the reference bonds whose particles carry conflicting cut sides."""
    a, b = lattice_bonds(lattice)
    codes = np.asarray(codes)
    severed = conflicting_packed(codes[a], codes[b]) if packed else conflicting(codes[a], codes[b], cuts)
    return a[severed], b[severed]


def bridge_report(nodes, weights, gradients, codes, *, digit_index=0, cuts=1, packed=False, reference_positions=None,
                  limit=20):
    """Code-0 stencil bridges of one cut digit (as scripts/analyze_cut_bridges.py, any spacing, recorded stencil)."""
    nodes = np.asarray(nodes, np.int64)
    valid = (np.asarray(weights) > 0) | (np.linalg.norm(np.asarray(gradients), axis=-1) > 0)
    codes = np.asarray(codes)
    digit = (decode_digits(codes, cuts)[:, digit_index].astype(np.int64) if packed
             else (codes//3**digit_index) % 3)
    has = {k: np.zeros(nodes.max()+1, bool) for k in (0, 1, 2)}
    for k in has:
        has[k][nodes[digit == k][valid[digit == k]]] = True
    zero = np.flatnonzero(digit == 0)
    touches = lambda k: (has[k][nodes[zero]] & valid[zero]).any(1)  # noqa: E731
    direct = zero[touches(1) & touches(2)]
    report = {"digit": int(digit_index), "digit_counts": {int(k): int(v) for k, v in zip(*np.unique(digit, return_counts=True))},
              "nodes_with_digits_0_1": int((has[0] & has[1]).sum()), "nodes_with_digits_0_2": int((has[0] & has[2]).sum()),
              "nodes_with_digits_1_2": int((has[1] & has[2]).sum()), "direct_code_zero_bridges": int(len(direct))}
    if reference_positions is not None:
        report["bridge_reference_xyz_mm"] = np.round(np.asarray(reference_positions)[direct][:limit]*1e3, 2).tolist()
    return report


def mass_check(task, recorded_masses, recorded_count=None):
    """Recorded particle masses vs the task sampler under the frozen v1 tolerance (cutting_tasks_v1 common)."""
    common = _v1()["common"]
    _, sampler, _ = task.sample_particles()
    recorded = np.asarray(recorded_masses, float)
    sampler = np.asarray(sampler, float)
    total, expected = float(recorded.sum()), float(sampler.sum())
    bound = max(common["mass_relative_tolerance"]*expected, common["mass_absolute_tolerance_kg"])
    count_ok = len(recorded) == len(sampler) and (recorded_count is None or int(recorded_count) == len(sampler))
    return {"recorded_total_mass_kg": total, "sampler_total_mass_kg": expected, "absolute_difference_kg": abs(total-expected),
            "allowed_difference_kg": bound, "recorded_particles": int(len(recorded)), "sampler_particles": int(len(sampler)),
            "solver_particles": None if recorded_count is None else int(recorded_count),
            "per_particle_max_abs_difference_kg": float(np.abs(recorded-sampler).max()) if len(recorded) == len(sampler) else None,
            "pass": bool(count_ok and abs(total-expected) <= bound),
            "tolerance_source": "cutting_tasks_v1.json common.mass_relative_tolerance / mass_absolute_tolerance_kg"}


def component_masses(masses, labels, count):
    masses = np.asarray(masses, float)
    per = np.bincount(labels, weights=masses, minlength=count)
    order = np.argsort(per)[::-1]
    return {"component_mass_kg": per[order].tolist(), "component_mass_fraction": (per[order]/masses.sum()).tolist(),
            "component_ids_by_mass": order.tolist(), "total_mass_kg": float(masses.sum())}


# ----------------------------------------------------------------------------- per-step summaries

def contact_window_force(steps, phases, *, use="total", contact="blade"):
    """Mean/max |F| over the control steps of ``phases`` with tool-material contact.

    ``use``: 'total' (blade + obstacle boxes), 'blade' or 'grip'. ``contact``:
    'blade' (blade contact count > 0), 'grip' (grip obstacle reaction != 0) or
    'any' (either). This is the Phase F contact window replacing C4's all-steps mean.
    """
    rows = [s for s in steps if s["phase"] in phases]
    key = {"total": "total_force_world_N", "blade": "blade_force_world_N", "grip": "handle_force_local_N"}[use]

    def touching(s):
        blade = s["contact_statistics"][1] > 0
        grip = s.get("handle_force_local_N") is not None and float(np.linalg.norm(s["handle_force_local_N"])) > 0
        return {"blade": blade, "grip": grip, "any": blade or grip}[contact]

    window = [s for s in rows if touching(s)]
    out = {"phases": list(phases), "phase_steps": len(rows), "contact_steps": len(window), "force": use, "contact": contact}
    if not window:
        return {**out, "mean_N": 0., "max_N": 0.}
    values = np.linalg.norm([s[key] for s in window], axis=1)
    all_values = np.linalg.norm([s[key] if s[key] is not None else [0., 0., 0.] for s in rows], axis=1)
    return {**out, "mean_N": float(values.mean()), "max_N": float(values.max()),
            "all_phase_steps_mean_N": float(all_values.mean()),
            "first_contact_time_s": float(window[0]["time_s"]), "last_contact_time_s": float(window[-1]["time_s"])}


def plowing_summary(steps):
    """Law-8 plowing (last substep of each control step) over the armed steps with cutting motion."""
    armed = [s for s in steps if s["cut_enabled"]]
    moving = [s for s in armed if (s.get("cutting_segments_last_substep") or 0) > 0]
    segments = steps[0].get("edge_segments") if steps else None
    if not armed or segments is None:
        return {"armed_steps": len(armed), "status": "NOT_RUN"}
    plow = np.asarray([s.get("plowing_segments_last_substep") or 0 for s in armed], float)
    ok = np.asarray([s.get("cutting_segments_last_substep") or 0 for s in armed], float)
    return {"armed_steps": len(armed), "armed_steps_with_cutting_segments": len(moving),
            "armed_steps_with_plowing": int(np.count_nonzero(plow > 0)),
            "plowing_step_fraction_of_cutting_steps": float(np.count_nonzero(plow > 0)/len(moving)) if moving else None,
            "plowing_segment_fraction_of_cutting_segments": float(plow.sum()/ok.sum()) if ok.sum() else None,
            "edge_segments": int(segments),
            "scope": "segment_plow / segment_ok sampled after the last substep of each control step (law 8)"}


def stroke_summary(steps, phases=None):
    rows = [s for s in steps if phases is None or s["phase"] in phases]
    if not rows:
        return {"phases": None if phases is None else list(phases), "steps": 0}
    return {"phases": None if phases is None else list(phases), "steps": len(rows),
            "armed_steps": sum(bool(s["cut_enabled"]) for s in rows),
            "stroke_active_steps": sum(s["stroke"] != 0 for s in rows),
            "engaged_steps_last_substep": sum(bool(s["engaged"]) for s in rows),
            "any_motion_steps_last_substep": sum(bool(s["any_motion"]) for s in rows),
            "max_travel_since_engaged_m": float(max(s["travel_m"] for s in rows)),
            "cutting_segments_last_substep": [int(s.get("cutting_segments_last_substep") or 0) for s in rows],
            "stroke_per_step": [int(s["stroke"]) for s in rows],
            "labelled_before": None, "labelled_after": int(rows[-1]["labelled_count"])}


def swept_normal_report(steps, phases, *, segment_l_m=(-.015, 0., .015)):
    """Host-side law-3/4 swept-surface normal at edge positions l from consecutive control poses.

    Mirrors ``tlmpm_v4._swept_side``: with the segment motion m = v + w x (e l),
    n_s = (m x e)/|m x e| (sign toward the blade normal), falling back to the
    blade normal when |m x e|^2 <= 1e-20. Reported per step and per l as the angle
    between n_s and the blade normal. The solver uses the substep twist; this is
    the control-step view (v and w from consecutive control poses).
    """
    per_l = {float(l): [] for l in segment_l_m}
    previous = None
    for s in steps:
        pose = np.asarray(s["edge_pose_world"], float)
        if previous is not None and s["phase"] in phases:
            dt = s["time_s"]-previous[0]
            R0, R = Rotation.from_quat(previous[1][3:]), Rotation.from_quat(pose[3:])
            v = (pose[:3]-previous[1][:3])/dt
            w = (R*R0.inv()).as_rotvec()/dt
            e, n = R.apply([0., 1., 0.]), R.apply([0., 0., 1.])
            for l in per_l:
                m = v+np.cross(w, e*l)
                cross = np.cross(m, e)
                norm2 = float(cross@cross)
                if norm2 > 1e-20:
                    ns = cross/np.sqrt(norm2)
                    ns = ns if ns@n >= 0 else -ns
                    angle, fallback = float(np.degrees(np.arccos(np.clip(ns@n, -1., 1.)))), False
                else:
                    angle, fallback = 0., True
                per_l[l].append((angle, fallback, norm2, float(np.linalg.norm(m))))
        previous = (s["time_s"], pose)
    out = {"phases": list(phases), "steps": len(next(iter(per_l.values()))) if per_l else 0, "segments": {}}
    for l, rows in per_l.items():
        if not rows:
            continue
        angles = np.asarray([r[0] for r in rows])
        out["segments"][f"{l*1e3:+.1f}mm"] = {
            "fallback_steps": int(sum(r[1] for r in rows)), "max_normal_angle_deg": float(angles.max()),
            "mean_normal_angle_deg": float(angles.mean()), "min_cross_norm2": float(min(r[2] for r in rows)),
            "max_segment_speed_m_s": float(max(r[3] for r in rows))}
    return out


# ----------------------------------------------------------------------------- multi-piece diagnostics

def cross_component_proximity(positions, labels, a, b, distance_m):
    """Particle pairs of components a and b closer than ``distance_m`` at one frame (and the minimum distance)."""
    from scipy.spatial import cKDTree
    positions = np.asarray(positions, float)
    pa, pb = positions[labels == a], positions[labels == b]
    if not len(pa) or not len(pb):
        return {"pairs_within": 0, "minimum_distance_m": None}
    tree = cKDTree(pb)
    distance, _ = tree.query(pa, k=1)
    pairs = tree.query_ball_point(pa, distance_m, return_length=True)
    return {"pairs_within": int(np.sum(pairs)), "particles_a_within": int(np.count_nonzero(pairs)),
            "minimum_distance_m": float(distance.min()), "distance_m": float(distance_m)}


def digit_persistence(label_steps, label_old, label_new, cuts, digit_end_steps, *, packed=True):
    """Changes of digit k recorded at control steps after ``digit_end_steps[k]`` (per-step label records)."""
    label_steps = np.asarray(label_steps, np.int64)
    old, new = np.asarray(label_old), np.asarray(label_new)
    if packed:
        d_old, d_new = decode_digits(old, cuts), decode_digits(new, cuts)
    else:
        d_old = np.stack([(old//3**k) % 3 for k in range(cuts)], 1)
        d_new = np.stack([(new//3**k) % 3 for k in range(cuts)], 1)
    out = {}
    for k, end in digit_end_steps.items():
        changed = d_old[:, k] != d_new[:, k]
        late = changed & (label_steps > end)
        rewrite = changed & (d_old[:, k] != 0)
        out[str(k)] = {"end_step": int(end), "changes_total": int(changed.sum()), "changes_after_end": int(late.sum()),
                       "nonzero_digit_rewrites": int(rewrite.sum())}
    return out


def stability_summary(min_det, max_speed, finite):
    min_det, max_speed = np.asarray(min_det, float), np.asarray(max_speed, float)
    return {"all_positions_velocities_finite": bool(np.all(finite)), "minimum_det_F": float(min_det.min()),
            "inverted_steps": int(np.count_nonzero(min_det <= 0)), "maximum_particle_speed_m_s": float(max_speed.max()),
            "stable": bool(np.all(finite) and min_det.min() > 0),
            "scope": "Checked after every control step on the device state (positions, velocities, deformation gradient)"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

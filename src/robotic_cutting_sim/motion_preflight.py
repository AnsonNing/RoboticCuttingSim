"""Geometric motion preflight for configured tasks and the physical return-to-home (WP-11 Phase D).

Reuses the WP-11 C2 checker (``scripts/preflight_wp11_orientations.py``): the
runtime's Newton analytic IK (``DualFrankaController`` with the profile
``hand_from_edge``) and MuJoCo signed distances on the official Franka
colliders, finger visual hulls, the profile handle/blade hulls, both v5 wrist
D405 boxes, the D455, the task's board/riser/table boxes, the declared workpiece
base box and protrusion hull and the parked Franka 1. Geometry only: no
dynamics, no material, no grasp holding; the workpiece is the declared (uncut)
geometry.

Blade-workpiece modes per sample (other pairs use the C2 thresholds unchanged):

* ``free``: the C2 free-space clearance (2 mm);
* ``contact_free``: no penetration (clearance >= 0), e.g. a PRE_CUT edge 0.2 mm
  above the plane or a lift out of a near-contact pose;
* ``contact_allowed``: armed cutting and the blade-in-material phases (CUT,
  SKIM, HOLD, WITHDRAW, LOWER, PUSH, PUSH_HOLD, RELEASE, RETRACT), as C2 allows.

Needs ``newton``, ``warp``, ``mujoco`` and ``pxr`` (``.venv``); the Isaac runtime
calls it in a separate ``.venv`` process (``scripts/wp11_motion_preflight.py``).
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
C2_SCRIPT = ROOT/"scripts/preflight_wp11_orientations.py"
CLI_SCRIPT = ROOT/"scripts/wp11_motion_preflight.py"
CONTACT_ALLOWED_PHASES = ("CUT", "SKIM", "HOLD", "WITHDRAW", "LOWER", "PUSH", "PUSH_HOLD", "RELEASE", "RETRACT")
CONTACT_FREE_PHASES = ("PRE_CUT",)
MODES = ("free", "contact_free", "contact_allowed")
_C2 = None


def c2_module():
    """The C2 preflight script as a module (loaded once)."""
    global _C2
    if _C2 is None:
        spec = importlib.util.spec_from_file_location("preflight_wp11_orientations", C2_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _C2 = module
    return _C2


def phase_mode(label, armed):
    """Blade-workpiece mode of a plan sample from its phase label and armed digit."""
    phase = str(label).split(":", 1)[0]
    if armed or phase in CONTACT_ALLOWED_PHASES:
        return "contact_allowed"
    if phase in CONTACT_FREE_PHASES:
        return "contact_free"
    return "free"


def blade_workpiece_threshold(mode):
    c2 = c2_module()
    return {"free": c2.THRESHOLDS["blade_free_workpiece_clearance_m"], "contact_free": 0., "contact_allowed": None}[mode]


class MotionChecker:
    """IK, FK and the C2 collision check for one task and its tool profile."""

    def __init__(self, task):
        c2 = c2_module()
        self.task, self.profile_id = task, task.tool_profile_id
        start = time.perf_counter()
        self.kin = c2.Kinematics(self.profile_id)
        self.geo = c2.GeometryModel(self.profile_id, task)
        self.build_wall_s = time.perf_counter()-start
        self.thresholds = dict(c2.THRESHOLDS)

    @property
    def joint_lower(self):
        return self.kin.lower

    @property
    def joint_upper(self):
        return self.kin.upper

    def edge_fk(self, q7):
        return self.kin.edge_pose(np.asarray(q7, float))

    def solve_ik(self, pose, seed, strict=False):
        q, info = self.kin.solve(np.asarray(pose, float), np.asarray(seed, float), strict)
        return q, {key: (float(value) if isinstance(value, (float, np.floating)) else value) for key, value in info.items()}

    def check_joint_path(self, qs, modes, labels=None):
        """Worst slack per check over joint samples; PASS iff every check passes."""
        c2 = c2_module()
        qs = np.asarray(qs, float)
        if len(qs) != len(modes) or any(mode not in MODES for mode in modes):
            raise ValueError("One known blade-workpiece mode per joint sample is required")
        labels = list(labels) if labels is not None else [None]*len(qs)
        lower, upper = self.kin.lower, self.kin.upper
        worst = {}
        # WP-12 flat-board workcell (ADR 0017): while the blade may touch material (contact_free/contact_allowed)
        # it may reach the rigid board top but never enter it; free-space samples keep the C2 1 mm clearance.
        record = getattr(self.task, "flat_board_record", None)
        board_contact = None if record is None else float(record["preflight"]["blade_board_contact_min_distance_m"])

        def note(check, value, threshold, sample, **extra):
            slack = value-threshold
            if check not in worst or slack < worst[check]["slack"]:
                worst[check] = {"slack": float(slack), "value": float(value), "threshold": float(threshold),
                                "sample": int(sample), "label": labels[sample], **extra}
        start = time.perf_counter()
        for i, (q, mode) in enumerate(zip(qs, modes)):
            margin = float(np.minimum(q-lower, upper-q).min())
            note("joint_limit_margin", margin, self.thresholds["joint_margin_rad"], i,
                 joint=int(np.argmin(np.minimum(q-lower, upper-q)))+1)
            if i:
                note("joint_continuity", -float(np.abs(q-qs[i-1]).max()), -self.thresholds["joint_step_rad"], i)
            self.geo.set_q(q)
            for (subject, target), (dist, g, h) in self.geo.evaluate().items():
                if subject == "blade" and target == "workpiece":
                    threshold = blade_workpiece_threshold(mode)
                elif subject == "blade" and target == "board" and board_contact is not None and mode != "free":
                    threshold = board_contact
                else:
                    threshold = c2.pair_threshold(subject, target, False)
                if threshold is None:
                    continue
                pair = [self.geo.geoms[g]["name"], self.geo.geoms[h]["name"]]
                for check in c2.checks_for(subject, target):
                    note(check, dist, threshold, i, pair=pair, mode=mode if subject == "blade" else None)
        failing = sorted(name for name, row in worst.items() if row["slack"] < 0)
        binding = min(worst, key=lambda name: worst[name]["slack"]) if worst else None
        return {"status": "PASS" if not failing else "FAIL", "failing": failing,
                "binding": failing[0] if failing else binding, "worst": worst, "samples": int(len(qs)),
                "wall_s": time.perf_counter()-start, "checker": "C2 GeometryModel (scripts/preflight_wp11_orientations.py)",
                "thresholds": self.thresholds, "blade_workpiece_modes": {mode: int(sum(m == mode for m in modes)) for mode in MODES},
                "scope": "Geometry only; declared (uncut) workpiece; knife at the nominal grasp; no dynamics or material"}


def plan_samples(task, *, until_phase=None, max_translation_m=.002, max_rotation_deg=1.):
    """Dense WORLD edge samples of the task reference schedule (<= 2 mm, <= 1 deg), with labels and armed flags."""
    from .episode_lifecycle import slerp_poses
    knots = np.asarray(task.edge_position_knots_local_txyz, float)
    labels, cuts = list(task.reference_phase_labels), list(task.reference_cut_indices)
    stop = len(knots)
    if until_phase is not None:
        hits = [i for i, label in enumerate(labels) if label.split(":", 1)[0] == until_phase]
        if not hits:
            raise ValueError(f"The plan has no phase {until_phase!r}")
        stop = hits[-1]+2   # through the end pose of the last motion of that phase
    poses = [np.asarray(task.reference_edge_pose_world(float(t)), float) for t in knots[:stop, 0]]
    out, out_labels, out_armed = [poses[0]], [labels[0]], [cuts[0] >= 0]
    for i in range(1, len(poses)):
        a, b = out[-1], poses[i]
        distance = np.linalg.norm(b[:3]-a[:3])
        angle = np.degrees((Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(b[3:])).magnitude())
        if distance < 1e-9 and angle < 1e-6:
            continue
        n = max(1, int(np.ceil(distance/max_translation_m-1e-9)), int(np.ceil(angle/max_rotation_deg-1e-9)))
        for pose in slerp_poses(a, b, n):
            out.append(pose)
            out_labels.append(labels[i-1])
            out_armed.append(cuts[i-1] >= 0)
    return np.asarray(out), out_labels, out_armed


def preflight_plan(task, q_home, *, until_phase=None, checker=None):
    """IK chain from q_home over the task's dense reference plan, then the joint-path check."""
    checker = checker or MotionChecker(task)
    start = time.perf_counter()
    poses, labels, armed = plan_samples(task, until_phase=until_phase)
    th = checker.thresholds
    qs, seed, failure = [], np.asarray(q_home, float), None
    ik_worst = {"position_error_m": 0., "rotation_error_rad": 0.}
    for i, (pose, flag) in enumerate(zip(poses, armed)):
        q, info = checker.solve_ik(pose, seed, flag)
        tol_p = th["ik_active_position_m"] if flag else th["ik_position_m"]
        tol_r = th["ik_active_rotation_rad"] if flag else th["ik_rotation_rad"]
        ik_worst = {key: max(ik_worst[key], float(info[key])) for key in ik_worst}
        if info["position_error_m"] > tol_p or info["rotation_error_rad"] > tol_r:
            failure = {"sample": i, "label": labels[i], "ik_info": info, "position_tolerance_m": tol_p,
                       "rotation_tolerance_rad": tol_r}
            break
        qs.append(q)
        seed = q
    receipt = {"samples": int(len(poses)), "until_phase": until_phase, "ik_worst": ik_worst,
               "phases": sorted(set(label.split(":", 1)[0] for label in labels)),
               "q_home": list(map(float, q_home)), "home_edge_fk_world_xyzw": checker.edge_fk(q_home).tolist(),
               "reference_t0_world_xyzw": np.asarray(task.reference_edge_pose_world(0.)).tolist()}
    if failure is not None:
        return {**receipt, "status": "FAIL", "reason": "IK_FAILURE", "ik_failure": failure,
                "wall_s": time.perf_counter()-start}
    modes = [phase_mode(label, flag) for label, flag in zip(labels, armed)]
    path = checker.check_joint_path(qs, modes, labels)
    by_phase = {}
    for label, q in zip(labels, qs):
        by_phase.setdefault(label.split(":", 1)[0], 0)
        by_phase[label.split(":", 1)[0]] += 1
    return {**receipt, "status": path["status"], "reason": None if path["status"] == "PASS" else "SCENE_INFEASIBLE",
            "path": path, "samples_by_phase": by_phase, "q_path_endpoints": [qs[0].tolist(), qs[-1].tolist()],
            "q_until_end": qs[-1].tolist(), "wall_s": time.perf_counter()-start}


def plan_return_home_for_task(task, q_start, *, config=None, checker=None):
    """``episode_lifecycle.plan_return_home`` with the real IK/FK and the C2 checker."""
    from .episode_lifecycle import plan_return_home
    home = task.home
    if home is None:
        raise ValueError("The task's scene declares no HOME")
    checker = checker or MotionChecker(task)
    plan = plan_return_home(q_start, home["joint_positions"], edge_fk=checker.edge_fk,
                            solve_ik=lambda pose, seed: checker.solve_ik(pose, seed, False),
                            control_dt=task.control_dt_s, joint_lower=checker.joint_lower, joint_upper=checker.joint_upper,
                            checker=lambda qs, modes: checker.check_joint_path(qs, modes), config=config)
    return plan


# ---------------------------------------------------------------- separate-process entry (Isaac runtime)

def venv_python():
    return ROOT/".venv/bin/python"


def run_cli(mode, payload, *, timeout_s=900, log_dir=None, **arrays):
    """Run ``scripts/wp11_motion_preflight.py <mode>`` in the CPU ``.venv`` and return its JSON receipt.

    The Isaac Sim process never imports MuJoCo/Newton collision models; the
    request and the receipt are files (kept in ``log_dir`` when given).
    """
    directory = Path(log_dir) if log_dir is not None else Path(tempfile.mkdtemp(prefix="wp11_preflight_"))
    directory.mkdir(parents=True, exist_ok=True)
    stamp = f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.monotonic_ns() % 10**6}"
    request, out = directory/f"{stamp}_request.json", directory/f"{stamp}_receipt.json"
    request.write_text(json.dumps({"mode": mode, "payload": payload,
                                   **{k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v for k, v in arrays.items()}}))
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH")
           and not key.startswith(("CARB_", "OMNI_", "EXP_PATH", "ISAAC_"))}
    command = [str(venv_python()), "-I", str(CLI_SCRIPT), "--request", str(request), "--out", str(out)]
    start = time.monotonic()
    completed = subprocess.run(command, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout_s)
    (directory/f"{stamp}_log.txt").write_text(completed.stdout[-20000:]+"\n--- stderr ---\n"+completed.stderr[-20000:])
    if completed.returncode != 0 or not out.exists():
        return {"status": "ERROR", "reason": "PREFLIGHT_PROCESS_FAILED", "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:], "command": command, "wall_s": time.monotonic()-start}
    receipt = json.loads(out.read_text())
    receipt["process"] = {"command": command, "wall_s": time.monotonic()-start, "request": str(request), "receipt": str(out)}
    return receipt

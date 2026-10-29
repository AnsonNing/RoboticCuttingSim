"""Scripted-rollout reference trajectories and behavioural checks (WP-11 Phase A, CPU).

Why this exists
---------------
The v1/path_v2 ``WaypointPolicy`` rollouts were pinned by the SHA-256 of their
raw float64 action bytes. Those hashes are not environment independent: the
CI runner (OpenBLAS ``Haswell`` AVX2 kernel) and the development machine
(OpenBLAS ``SkylakeX`` AVX-512 kernel) disagree by one ulp in ``cblas_ddot``
on 3-vectors (``v @ w`` and ``np.linalg.norm``), with identical Python, NumPy
2.5.4 and SciPy 1.18.1. The first difference appears at step 40 of the sawing
rollout as 2.7e-20 m in an action component; over all rollouts the largest
difference is 8.7e-19 m in an action and 3.7e-18 m in the edge position.
Waypoint switching, cut flags and termination are identical. This is a
numeric-environment difference, not a behaviour difference
(docs/reports/WP11_PHASE_A_CI.md).

What is checked instead
-----------------------
1. The stored reference (``fixtures/scripted_rollout_reference_v1.json``) is
   exactly the historically pinned trajectory: its decoded action bytes hash
   to the original digests in ``HISTORIC_ROLLOUT_SHA256``. That comparison is
   on stored data, so it is environment independent.
2. The live rollout matches the reference: cut flag, cut index and waypoint
   index sequence exactly; continuous action components within
   ``rollout_tolerance(steps)`` (absolute, rtol = 0).
3. Behaviour (``check_behaviour``): action bounds, waypoint progress and
   reach accuracy, trajectory (cross-track) error, termination.

Tolerance
---------
``rollout_tolerance(N) = N * OPS_PER_STEP * eps`` with ``eps`` = 2.22e-16
(float64) and ``OPS_PER_STEP = 16``. Every quantity compared is at most
1 in magnitude in SI units (edge |x| <= 0.11 m, translation <= 2 mm,
rotation <= 0.18 rad). One control step has about 16 rounding-sensitive
operations (a few 3-vector dot products and norms, quaternion products and
rotation-vector conversions) whose results may differ by at most one ulp of
a quantity <= 1 between BLAS/libm implementations. The controllers do not
amplify such differences: v1 commands ``target - edge`` (unit gain, so an
earlier position error is removed rather than integrated) and path_v2 uses
cross-track gain 0.5. A linear worst case over N steps is therefore
N * 16 * eps: 2.1e-12 for 600 steps and 2.8e-12 for 800 steps. That is
about 1e6 times the observed differences and about 1e8 times below the
smallest behavioural scale (the 0.3 mm position tolerance of path_v2). A
behaviour change, such as a waypoint switched one step earlier, changes an
action by up to 2 mm and is reported, as is any change of a discrete column.

Regenerating
------------
``.venv/bin/python tests/rollout_reference.py OUT.json`` writes a new
reference only if every rollout reproduces its historic digest bit for bit.
It refuses to overwrite an existing file.
"""
import base64
import hashlib
import json
import platform
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.policies import WaypointPolicy, controller_profile, skim, slice_push, through_cut
from robotic_cutting_sim.task_scenarios import configure_task
from robotic_cutting_sim.tasks import make_task

FIXTURE = Path(__file__).resolve().parent/"fixtures"/"scripted_rollout_reference_v1.json"
FORMAT = "scripted_rollout_reference_v1"
SCENARIO = "blademaster_workcell_v1"
COLUMNS = ("dx_m", "dy_m", "dz_m", "rx_rad", "ry_rad", "rz_rad", "cut_enabled", "cut_index")
EPS = float(np.finfo(np.float64).eps)
OPS_PER_STEP = 16

# Historic pins. CUT-01-R/*: tests/test_blademaster_demo.py ROLLOUT_SHA, pinned before Phase 2A (59e4efb tree).
# v1|path_v2/*: tests/test_wrist_depth_survey.py PINNED[".../rollout"], computed at 9b6df5f (before path_v2_vision).
HISTORIC_ROLLOUT_SHA256 = {
    "CUT-01-R/default": "be1ed8f9da6b3c73d417000514ea797d8c77a09f3718ad2876552ae156c1e0af",
    "CUT-01-R/saw3mm_tilt10": "6a4c5e2a900213a16fcd03d895e4e6b1076d79e965a15d22f06294e70aba7c03",
    "v1/SLICE-PUSH-01": "75b56407f7f1a3911f5ace4c61aa3e5cff82917068b1018ed8abd9195ce4ea2f",
    "v1/TRIM-01-R": "ccefd5e6575e294a141007aaf90579954e9c17a549adc5d80134931619212ecc",
    "v1/CUT-01-R": "7533ec98ba3cbead484af145161ff549b185014eaeb2b482bbdd34cc5c8fc556",
    "path_v2/SLICE-PUSH-01": "6553b64edf19f69af09b62c11ff7fb2dcd6680c1d2baf3840adbf044c17912e9",
    "path_v2/TRIM-01-R": "bc52ced3464d687c601ece0616f24ede6486b302bb62df53e17039ab58bdf46f",
    "path_v2/CUT-01-R": "937346d624fe116e4eb59549c868666982afb49a57f3a6a92a474596fbf7d34b",
}
BUILDERS = {"SLICE-PUSH-01": slice_push, "TRIM-01-R": skim, "CUT-01-R": through_cut}


def rollout_tolerance(steps):
    """Absolute cross-environment tolerance (m, rad) of an ``steps``-step rollout (module docstring)."""
    return steps*OPS_PER_STEP*EPS


def make_case(name):
    """(policy, task, start edge pose, steps) of a named reference rollout."""
    if name.startswith("CUT-01-R/"):
        # tests/test_blademaster_demo.py: unconfigured CUT-01-R, start 10 mm +x / 20 mm +z off the reference edge.
        task = make_task("CUT-01-R")
        kwargs = {"default": {}, "saw3mm_tilt10": {"saw_amplitude_m": .003, "tilt_deg": 10.}}[name.split("/")[1]]
        edge = np.asarray(task.reference_edge_pose_world(0.), float)
        return WaypointPolicy(task, through_cut(task, **kwargs)), task, np.r_[edge[:3]+(.01, 0., .02), edge[3:]], 600
    # tests/test_wrist_depth_survey.py: controller profile on the blademaster_workcell_v1 task, start at plan[0].
    profile, task_id = name.split("/")
    policy_options, plan_options = controller_profile(profile)
    task = configure_task(make_task(task_id), scenario=SCENARIO)
    plan = BUILDERS[task_id](task, **plan_options)
    return WaypointPolicy(task, plan, **policy_options), task, np.asarray(plan[0].pose, float), 800


def ideal_rollout(policy, task, start, steps):
    """Ideal integrator: the observed edge moves by each commanded world delta.

    Returns actions (steps x 8, ``COLUMNS``), the waypoint index after each call,
    the edge pose observed at each step (steps x 7) and ``policy.last_tracking``.
    """
    edge, actions, index, edges, tracking = np.asarray(start, float), [], [], [], []
    for k in range(steps):
        action = policy({"proprio": {"tool_pose": edge.copy()}, "timestamp_s": k*task.control_dt_s})
        actions.append(np.r_[action["target"], float(action["cut_enabled"]), action.get("cut_index", 0)])
        index.append(policy.index)
        edges.append(edge.copy())
        tracking.append(policy.last_tracking)
        edge = np.r_[edge[:3]+action["target"][:3],
                     (Rotation.from_rotvec(action["target"][3:])*Rotation.from_quat(edge[3:])).as_quat()]
    return dict(actions=np.asarray(actions), index=np.asarray(index), edges=np.asarray(edges), tracking=tracking)


def run_lengths(values):
    """[[value, count], ...] of consecutive equal integers."""
    runs = []
    for value in np.asarray(values).tolist():
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return runs


def load_reference(path=FIXTURE):
    """{name: {"actions": array, "index_runs": runs, ...}}; checks format and each historic digest."""
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    out = {}
    for name, entry in data["rollouts"].items():
        raw = zlib.decompress(base64.b64decode(entry["actions_zlib_b64"]))
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["actions_sha256"] or digest != HISTORIC_ROLLOUT_SHA256[name]:
            raise ValueError(f"{path}: {name} actions do not hash to the historic pin")
        actions = np.frombuffer(raw, dtype="<f8").reshape(entry["shape"])
        out[name] = dict(entry, actions=actions)
    return out


def compare_to_reference(live, reference, tol):
    """Discrete columns and waypoint sequence exactly, continuous columns within ``tol``; returns max |delta|."""
    actions, expected = live["actions"], reference["actions"]
    assert actions.shape == expected.shape
    assert np.array_equal(actions[:, 6:], expected[:, 6:]), "cut_enabled / cut_index sequence changed"
    assert run_lengths(live["index"]) == reference["waypoint_index_runs"], "waypoint progression changed"
    delta = np.abs(actions[:, :6]-expected[:, :6])
    worst = int(np.argmax(delta.max(axis=1)))
    assert np.all(np.isfinite(actions)) and delta.max() <= tol, \
        f"continuous action differs by {delta.max():.3e} > {tol:.3e} first at step {int(np.argmax(delta.max(1) > tol))}"
    return float(delta.max()), worst


def _distance_to_segment(point, start, end):
    direction = end-start
    length = float(np.linalg.norm(direction))
    s = np.clip((point-start)@direction/length, 0., length) if length > 0 else 0.
    return float(np.linalg.norm(point-(start+(direction/length if length > 0 else 0.)*s)))


def check_behaviour(live, policy, task, tol):
    """Action bounds, waypoint progress, reach accuracy, trajectory error and termination of an ideal rollout."""
    waypoints, limits, dt = policy.waypoints, policy.limits, task.control_dt_s
    actions, index, edges, tracking = live["actions"], live["index"], live["edges"], live["tracking"]
    n = len(waypoints)
    assert np.all(np.isfinite(actions))
    # Termination: every waypoint is reached inside the rollout; afterwards zero motion, cutting off.
    assert index[-1] == n and policy.done and policy.phase == "done"
    done = int(np.argmax(index == n))
    assert np.all(actions[done:, :6] == 0.) and np.all(actions[done:, 6] == 0.)
    # Waypoint progress: monotone; every passed waypoint was reached within the controller tolerances.
    assert np.all(np.diff(index) >= 0) and index[0] >= 0
    previous = 0
    for k in np.nonzero(np.diff(np.r_[0, index]))[0]:
        for j in range(previous, index[k]):
            target = np.asarray(waypoints[j].pose, float)
            assert np.linalg.norm(edges[k, :3]-target[:3]) <= policy.position_tolerance_m+tol, (k, j)
            rotation = (Rotation.from_quat(target[3:])*Rotation.from_quat(edges[k, 3:]).inv()).magnitude()
            assert rotation <= policy.rotation_tolerance_rad+tol, (k, j)
            if waypoints[j].hold_s > 0:
                assert int(np.sum(index == j)) >= round(waypoints[j].hold_s/dt), ("hold too short", j)
        previous = index[k]
    for k in range(done):
        waypoint = waypoints[index[k]]
        translation, rotation = actions[k, :3], actions[k, 3:6]
        # Cut flags follow the active waypoint.
        assert actions[k, 6] == float(waypoint.cut) and actions[k, 7] == waypoint.cut_index
        # Action bounds: per-step translation limit and waypoint speed; rotation is a proper rotation vector.
        step = limits.max_translation_m if waypoint.speed_m_s is None else min(limits.max_translation_m,
                                                                                 waypoint.speed_m_s*dt)
        assert np.linalg.norm(rotation) <= np.pi+tol
        if tracking[k] is not None:
            assert np.linalg.norm(translation) <= min(limits.max_translation_m,
                                                      step+tracking[k]["cross_track_command_m"])+tol
            # Trajectory error: on an ideal plant path_v2 stays on the planned segment (to rounding).
            assert tracking[k]["cross_track_error_m"] <= tol, (k, tracking[k])
            continue
        start = np.asarray(waypoints[index[k]-1].pose[:3], float) if index[k] > 0 else None
        if waypoint.saw_amplitude_m > 0:
            along = Rotation.from_quat(np.asarray(waypoint.pose[3:], float)).apply([0., 1., 0.])
            assert abs(translation@along) <= limits.max_translation_m+tol
            assert np.linalg.norm(translation-(translation@along)*along) <= step+tol
            if waypoint.cut and start is not None:
                # Off the planned cut segment only along the edge line (the saw stroke).
                lateral = (edges[k, :3]-start)@along
                assert _distance_to_segment(edges[k, :3]-lateral*along, start,
                                            np.asarray(waypoint.pose[:3], float)) <= policy.position_tolerance_m+tol
        else:
            assert np.linalg.norm(translation) <= step+tol
            if waypoint.cut and start is not None:
                # Trajectory error: v1 moves straight from where the previous waypoint was reached.
                assert _distance_to_segment(edges[k, :3], start,
                                            np.asarray(waypoint.pose[:3], float)) <= policy.position_tolerance_m+tol
    return done


def _environment():
    import scipy
    info = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                machine=platform.machine(), processor=platform.processor())
    import os
    import subprocess
    # OpenBLAS prints the runtime-selected kernel ("Core: SkylakeX") at import with OPENBLAS_VERBOSE=2.
    probe = subprocess.run([sys.executable, "-c", "import numpy"], capture_output=True, text=True,
                           env={**os.environ, "OPENBLAS_VERBOSE": "2"})
    info["openblas_core"] = (probe.stdout+probe.stderr).strip()
    info["openblas_coretype_override"] = os.environ.get("OPENBLAS_CORETYPE")
    return info


def main(out):
    out = Path(out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rollouts = {}
    for name, historic in HISTORIC_ROLLOUT_SHA256.items():
        policy, task, start, steps = make_case(name)
        live = ideal_rollout(policy, task, start, steps)
        raw = np.ascontiguousarray(live["actions"], dtype="<f8").tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != historic:
            raise SystemExit(f"{name}: rollout {digest} does not reproduce the historic pin {historic}; "
                             "regenerate only in an environment that reproduces it bit for bit")
        rollouts[name] = dict(steps=steps, shape=list(live["actions"].shape), dtype="<f8", columns=list(COLUMNS),
                              actions_sha256=digest, waypoint_index_runs=run_lengths(live["index"]),
                              done_step=int(np.argmax(live["index"] == len(policy.waypoints))),
                              actions_zlib_b64=base64.b64encode(zlib.compress(raw, 9)).decode())
    data = dict(format=FORMAT, generator="tests/rollout_reference.py", environment=_environment(),
                description="Ideal-integrator WaypointPolicy rollouts whose action bytes hash to the historic pins; "
                            "compared with tolerance steps*16*eps (tests/rollout_reference.py).",
                rollouts=rollouts)
    out.write_text(json.dumps(data, indent=1)+"\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])

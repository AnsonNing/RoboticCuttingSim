# Dynamic dual-Franka control — 2026-10-28

`robot.py` adds a real standalone Newton 1.6.1 articulated robot testbed. It is
independent of the existing blade-only MPM environment. Both Pandas have seven
dynamic arm joints and two finger joints; the second arm holds a park target
turned toward +Y, away from the cutting arm. There is no policy or training code.

The first hand carries a rigid box knife, mounted 180 mm along hand +Z. A shape
on the hand link is an exactly fixed attachment, including its mass and inertia;
there is no independently commanded free knife in this robot testbed. Published
knife pose comes from measured joint state and forward kinematics. SI units,
world Z up and XYZW quaternions apply throughout.

## Implementation and sources

The already cached [official Isaac 6.2 Franka USD](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda)
provides inertias, joint definitions and collision meshes. Newton's USD importer
loads the `Physics=mujoco` and `Colliders=convex_hulls` variants. Self-collision is
enabled through session-layer overrides; cached vendor files remain unchanged.
No vendor asset is committed. The existing NVIDIA asset terms in
[ASSET_SOURCES](../assets/ASSET_SOURCES.md) still apply.

Control uses position drives, explicit gravity compensation, Panda effort limits
(87 N m proximal, 12 N m wrist), and Newton's analytic-Jacobian damped IK with
position, rotation and joint-limit objectives. These gains, armatures and the
knife geometry are provisional simulation settings. References are the pinned
[Newton robot USD example](https://github.com/newton-physics/newton/blob/v1.6.1/newton/examples/robot/example_robot_ur10.py),
[IK example](https://github.com/newton-physics/newton/blob/v1.6.1/newton/examples/ik/example_ik_franka.py)
and [MuJoCo solver](https://github.com/newton-physics/newton/blob/v1.6.1/newton/_src/solvers/mujoco/solver_mujoco.py).
Runtime dependencies are Newton/Warp from the existing lock plus exact MuJoCo
3.12.0, MuJoCo Warp 3.12.0 and Newton USD schemas 0.6.0. Newton and both MuJoCo
packages use Apache-2.0; Newton USD schemas use Apache-2.0. They are installed as
external packages and are not vendored.

The CPU MuJoCo backend is the validated default for this small two-arm scene.
The optional GPU backend runs but did not meet the frozen reset tolerance in the
360-step test; its numerical failures are retained below. This does not change
the existing working GPU MPM backend.

## Reproduce

```bash
.venv/bin/python -m pip install -r requirements-robot.lock
# Fetch official assets once if the workcell cache has not been created:
.venv/bin/python -m robotic_cutting_sim.cli scene --preset workcell --output-dir artifacts/robot_assets_fresh
OMP_NUM_THREADS=4 .venv/bin/python scripts/check_robot_control.py --output-dir artifacts/dual_franka_fresh
.venv/bin/python -m pytest tests/test_robot_contract.py -q
```

The probe refuses to overwrite output. It records measured joints/velocities,
all link poses, desired and measured knife poses, timestamps, exact package/code
versions, source hashes, contacts, timing, diagnostic wrench response and a full
geometry USD replay. `tracking.png` is an image from the measured trace. Native
solver state, including warm-start buffers, is cleared on reset.

```python
from robotic_cutting_sim.robot import DualFrankaController

robot = DualFrankaController()
target = [-.055, 0, .07, 1, 0, 0, 0]  # world knife center, not wrist
q, ik = robot.inverse_kinematics(target)
assert ik["converged"]
robot.reset(q)
state, info = robot.step_cartesian(target)
state, info = robot.step_joints(q)
# state: q[2,7], qdot[2,7], tool_pose[7], timestamp_s
```

The internal `tool_wrench_world` keyword accepts force then torque at the knife
center. The adapter shifts that wrench to the hand COM before articulated
dynamics. It is for physics/coupling diagnostics and never enters proprioception.
It is not a material-force source. Contact diagnostics are also a separate API.

## Actual validation

The canonical CPU receipt is `artifacts/dual_franka_cpu_006/result.json`, with
compact public evidence in [robot_control_validation.json](evidence/robot_control_validation.json).
The test uses 60 Hz control and eight rigid substeps (480 Hz), with a 6 s smooth
approach, 110 mm sweep and withdrawal. It contains no deformable material.

| Check | Measured result |
|---|---|
| Joint position dynamics | PASS: 0.05 rad command moves the actual joint; final error 3.10e-6 rad |
| Cartesian path | PASS: maximum knife position error 4.108 mm, RMS 2.621 mm; threshold 10 mm |
| Fixed knife | PASS: measured hand transform agrees with mounted knife transform |
| Parked second arm | PASS: maximum drift 2.47e-14 rad |
| Same reset repeat | PASS: maximum joint difference exactly 0 over two 60-step repeats |
| Reference path rigid contacts | PASS: zero contact candidates on the tested path |
| Non-adjacent self-collision | PASS: separate deterministic negative pose detects a 20.04 mm collision |
| Robot-table collision | PASS: separate negative pose detects a 20.77 mm wrist/table collision |
| Knife-table dynamic response | PASS: 38.47 N peak normal support, 0.114 mm maximum penetration |
| Applied diagnostic wrench | PASS: +20 N world X displaces the knife 10.24 mm in X and returns within 3.57 micrometers on unloading |
| Measured link USD replay | PASS: read-back link position discrepancy 0 |
| CPU contract tests | PASS: 10 tests; these are not substitutes for the physical probes |

The table response test requests knife center Z=2 mm while the table prevents
travel beyond the blade edge. Measured center settles at Z=4.960 mm. This
demonstrates reaction through the articulated rigid body system, not merely
collision detection. The self/table negative queries are static geometry checks;
only the knife-table response is a dynamic collision experiment.

The complete CPU probe, including construction, the 6 s path, reset checks,
separate collision experiments and wrench loading/unloading, took 10.287 s wall.
Mean path control costs were 7.28 ms IK plus 1.57 ms physics per tick. Process
peak RSS was 549.83 MiB; no owned GPU allocation appeared in its memory sample.
This is the standalone rigid work package's performance, not integrated MPM/RTX
throughput. [Measured trace image](evidence/robot_tracking.png).

Optional GPU command (raw failures must be inspected):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python scripts/check_robot_control.py --backend mujoco_warp --device cuda:0 --steps 360 --output-dir artifacts/dual_franka_warp_fresh
```

`dual_franka_warp_001` compressed the same trajectory into 2 s (120 steps),
causing 12.15 mm tracking error and **FAIL** against the 10 mm threshold. Its
139.2 s wall time includes first kernel compilation. `dual_franka_warp_002`
uses 360 steps/6 s: tracking passes at 4.108 mm, but reset repeat differs by
1.665e-6 rad, exceeding the unchanged 1e-6 rad criterion (**FAIL**). No GPU
collision-export pass is claimed. Warm GPU means were 26.58 ms physics and
10.88 ms IK per control step, slower than the CPU backend for this small scene.
These runs used the solver's default nondeterministic mode. A deterministic-mode
GPU configuration is not validated here.

Earlier failures are preserved. `franka_import_probe_001/control_smoke.log`
shows the official primitive proxies overlap between link1 and link3 by about
40 mm when self-collision is enabled. Switching to the official convex meshes
resolved that problem without suppressing non-adjacent self-collision. The
original `dual_franka_cpu_001` receipt failed the knife/replay consistency check
because CPU Warp `.numpy()` views were retained across reused buffers. Logging
now copies each frame, and published link poses are recomputed from final joint
state after the native integrator. Run 003 completed its checks but failed final
stdout JSON serialization; run 006 is the fully successful reproduction.

## Integration boundary

Actual robot articulation, joint/Cartesian control, fixed attachment and rigid
contact response are validated independently. The tested reference path is not a
collision-avoidance planner or a promise about arbitrary actions.

Franka-to-MPM stepping, material-contact forces, persistent physical cutting,
bidirectional material/robot coupling, Isaac Lab articulation and RTX D455 are
**NOT_RUN/BLOCKED** in this robot work package. An external diagnostic force input
and successful table contact do not establish those capabilities. The standalone
robot can provide measured proprioception to a future integrated environment;
matching knife poses and timestamps alone will establish only one-way coupling.

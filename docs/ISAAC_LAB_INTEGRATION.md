# Live Isaac Lab 3.0 integration

This runner subclasses the actual `isaaclab.envs.DirectRLEnv`, declares both
Franka articulations and the rigid table/tray on `InteractiveSceneCfg`, and
advances the active Lab `NewtonManager`. It does not construct or step a
standalone robot simulator. No policy training is performed.

The official v3.0.0 source is pinned at
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. Its internal Python distribution
versions (`isaaclab` 28.0.0, `isaaclab-newton` 9.1.4) are recorded separately
from the source release. Python is 3.12.12; Torch is 2.11.0+cu128; Newton is
1.6.1; Warp is 1.17.0; `usd-exchange` 3.0.1 is the only USD provider in this
environment. CUDA tensor allocation on RTX 5090/driver 570.211.01 succeeded.
The official Isaac Sim 6.2 Python installation instructions explicitly support
the cu128 Torch variant. The default Lab cu130 stack has different requirements.

```bash
bash scripts/bootstrap.sh --runtime isaac_lab --managed-python
.venv/bin/python scripts/run_logged.py artifacts/lab_d0_new.log -- \
  .deps/isaaclab-venv/bin/python scripts/run_isaac_lab_cutting.py \
  --output-dir artifacts/lab_d0_new --steps 400
```

Let bootstrap create the environment; do not precreate an empty venv. The
installer preserves and rejects an incomplete existing environment. Current
installation, source overrides and deployment checks are documented in
[PORTABLE_DEPLOYMENT](PORTABLE_DEPLOYMENT.md).

Output directories must be new. The runner exports a fresh static workcell and
reuses only its cameras and lights; material and robot animation are computed
live. The existing official asset cache is reused or fetched by the exporter.
Never install `usd-core` beside `usd-exchange` in this environment.

Lab owns the 2 ms articulated dynamics steps (ten per 20 ms environment step).
The actual measured hand pose composes with the fixed 180 mm / Rx(pi) blade
mount. That measured edge drives the retained TLMPM's 200 x 0.1 ms substeps.
Its equal-opposite impulse supplies a world hand wrench in the next 20 ms
interval, shifted to the hand COM at each rigid step. Force and material oracle
state are debug-only. The target belongs to `WorkpieceTarget` at local z=15 mm;
the edge clearance is 0.5 mm. Current world-authored TLMPM fixture support is
explicitly limited to an identity workpiece pose.

Kit-less camera output is labelled `live_isaac_lab_ideal_warp_rgbd`. Full current
robot geometry, the measured mounted wedge, fixture, and the current mechanical
component surfaces enter its ray tracing at the material timestamp. These
images are useful live debugging evidence, but cannot pass native D455 or full
Kit gates. The Kit path is documented separately.

## Executed probes

- `artifacts/isaac_lab_smoke_006`: actual Scene/Env/Task, two control steps,
  0.04 s, rigid runtime clock 0.04000000000000002 s, 3.665 s wall time.
- `artifacts/isaac_lab_camera_smoke_001`: the same live physics plus both
  ideal RGB-D views/global and five labelled videos; 0.04 s / 4.496 s wall.
- `artifacts/isaac_lab_d0_001`: 400 steps / 8 s / 193.169 s wall, native
  clock 7.999999999999341 s, two final mechanical components, 1.656 g chip.
  The original controller fails frozen ROB-02: active maximum 1.23558 mm;
  all-phase maximum 5.70398 mm. The actual substrate triangle mesh also fails
  TASK-02: peak residual 1.904 mm / p95 absolute plane error 1.709 mm.
- `artifacts/isaac_lab_d0_feedback_off_001`: matched 400-step no-feedback
  physics control, 64.580 s wall with cameras explicitly disabled. Contact
  reaction changes actual robot motion. Baseline interface-work residual is
  2.3093%; this is a delayed-coupling diagnostic, not an energy convergence claim.
- `artifacts/isaac_lab_d0_feedforward_001`: the optional
  `--velocity-feedforward` sends the finite difference of commanded IK joints
  as target velocities; gains, path, material and thresholds are unchanged.
  Actual active tracking maximum falls to 0.50725 mm and all-phase to 2.36897 mm,
  passing ROB-02. Global resolution is 1280x960; D455 ideal views remain 640x480.
  All 400 material states are recorded. Reset q/material/depth differences are
  exactly zero with identical valid masks. TASK-02 still fails (1.974 mm peak).
  The interface-work residual rises to 19.15%, and the original post-withdrawal
  cross-component recontact criterion fails; these negative results are retained.

`--reset-check` inspects the initial state after a same-process reset; a full
independent repeat is a separate requirement. The latest recorder copies and
hashes its wrapper and physical implementation at launch. Older source can be
recovered only when its bytes exactly match a recorded hash, with an explicit
after-launch recovery receipt. Full native body contact auditing now reads the
actual MuJoCo Warp contact distances at rigid substeps; reset refreshes its
position/contact cache without integrating, avoiding stale constructor contacts.

Failures are retained: `isaac_lab_smoke_001` used Lab's advertised MuJoCo CPU
option but hit Newton's unimplemented `get_max_contact_count`; 002 found an
incorrect flat hand-path assumption; 003–005 exposed the XYZW convention and
the need to explicitly write Lab's nonsingular reset defaults before IK.
Neither vendor source nor the driver was patched. The supported runner uses
the actual MJWarp GPU path. Metadata-copy warnings concern Kit-only UI fields
in the standalone USD provider and remain in the logs.

Official references: [Lab installation](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation),
[Newton integration](https://docs.nvidia.com/isaaclab/3.0/concepts/newton),
[Isaac Sim 6.2 Python installation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html).

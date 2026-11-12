# RoboticCuttingSim

A configurable robotic cutting research environment with live **Native Isaac Sim 6.2**,
two Franka Pandas and an end-held blade in a physical two-finger grasp. One continuous
specimen supports **original v7 for the validated Straight CUT research profile**,
with **v5 / Law 8 retained as the legacy default**. Persistent cut histories, multiple
cuts and RGB-D observations use separate scene, goal, controller and evaluation APIs.
Material parameters are provisional and uncalibrated; Isaac Lab parity is unverified.

[![Sequential removal of two different protrusions — actual Native multiview animation](docs/media/three_global_v1/two_bumps_v7_2_native.gif)](docs/media/three_global_v1/two_bumps_v7_2_native.mp4)

**Sequential removal of two different protrusions — real Native Isaac Sim, seed 42, 1×.**
The versioned experimental `tlmpm_v7_2` detaches the 9 mm dome and 5.5 mm asymmetric
cap in separate passes: **A/B 100%, mechanical bodies 1 → 2 → 3, substrate retained,
HOME PASS**. The movie shows **three fixed Global RGB views — LEFT, RIGHT and
OPPOSITE TOOL — plus arm_0 / arm_1 Wrist D405 RGB and Depth** at the same physics
timestamp. All external views are low-angle (17.12°), about 0.60 m away, with full
uncropped RGB. OPPOSITE TOOL looks back from the blade-tip side without sideways
yaw. Franka 1 remains at its normal park; both hand-local wrist mounts are unchanged.

Wrist depth is **optical Z in meters**, with a **0.07–0.50 m** blue→green→red
scale; black means invalid/out of range. A parked wrist may not see the specimen.
Global RGB is a Native camera replay of the exact saved live mesh and measured
robot/tool state; original wrist RGB-D is reused. These are synchronized state
views, not three new physics trials. Narrow Global stereo-depth controls remain
**BLOCKED**; no calibrated Global depth is claimed.
[Camera names, calibration, recording/replay commands and source receipts](docs/GLOBAL_CAMERA_VIEWS.md) ·
[Retained earlier gallery](https://github.com/AnsonNing/RoboticCuttingSim/blob/b9100bcd662bafcb81cefedae424856d047e9e72/docs/NATIVE_MULTIVIEW_GALLERY.md).
[Scene / goal, measured state arrays and retained failures](docs/reports/WP14_FOLLOWUP_RESULTS.md) ·
[Current v7 multiview movies](docs/NATIVE_MULTIVIEW_GALLERY.md).
This is one verified demo; single-bump kinematic T00 still fails, and general TRIM
support is not claimed. The validated original-v7 CUT profile remains separate.

## Native cutting gallery

These three v7 camera-engineering recordings show CUT, experimental TRIM and
four slices: each source has task PASS and HOME PASS. They retain all three
external RGB views and both actual wrist RGB/depth pairs, with 1 mm particles,
4 mm grid and 0.1 ms substeps. Seed 42, `particle_surface_v1`. These selected
recordings are separate from the frozen scientific denominator and do not
establish general TRIM or multi-cut support.

| Straight CUT · 1× | TRIM at rake 10° · 1× | Four slices · 3× preview |
| --- | --- | --- |
| [![Straight CUT — real Native animation](docs/media/three_global_v1/wp13_v7_cut.gif)](docs/media/three_global_v1/wp13_v7_cut.mp4) | [![TRIM — real Native animation](docs/media/three_global_v1/wp13_v7_trim.gif)](docs/media/three_global_v1/wp13_v7_trim.mp4) | [![Four slices — real Native animation](docs/media/three_global_v1/wp13_v7_multicut.gif)](docs/media/three_global_v1/wp13_v7_multicut.mp4) |
| Two pieces | 100% above-plane dome mass detached (experimental Native recording) | Five pieces; placement quality reported separately |

[Multiview provenance and exact capture timestamps](docs/evidence/three_global/media_001.json) ·
[WP-13 measurements and limitations](docs/reports/WP13_RESULTS.md) ·
[earlier WP-12 demonstrations](docs/reports/WP12_FINAL_VALIDATION.md).
New full clips retain every Global capture and synchronize both real wrist RGB/depth streams.
Irregular captures keep their exact presentation timestamps; previews have labelled speeds.
[Current v7 recordings and historical evidence entry](docs/NATIVE_MULTIVIEW_GALLERY.md).

## CUT research profile (WP-14)

**`v7_cut_research_v1` is a scoped Validated Research Profile using original
`tlmpm_v7`. v5 / Law 8 remains the default.** The selected CUT configurations
and predeclared Native evidence are documented in the
[research report](docs/reports/WP14_RESULTS.md).

[![Updated WP-14 actual Native research scene](docs/media/three_global_v1/wp14_walkthrough.gif)](docs/media/three_global_v1/wp14_walkthrough.mp4)

**Executable scene: three Global RGB views and two wrist RGB-D pairs, 1×.**
[Launch the standard or firm-offset CUT scene](docs/WP14_SCENES.md) ·
[actual three-view scene](docs/media/three_global_v1/scene_three_global.png) ·
[scene/video provenance](docs/evidence/three_global/scene_001.json).
This extra engineering replay does not add an independent adoption sample.

Standalone v7 recordings: [standard CUT](docs/media/three_global_v1/wp14_v7_standard_cut.mp4)
and [firm offset CUT](docs/media/three_global_v1/wp14_v7_firm_offset_cut.mp4)
(firm material, yaw −5°, cut at −4 mm), both at 1× simulation time.
[Native tracking, physical contacts and HOME](docs/evidence/wp14/native_001.json).

| Selection / family | Supported scope |
| --- | --- |
| Default v5 + Law 8 | Existing baseline, unchanged |
| `v7_cut_research_v1` / Straight CUT | Four validated instances; constrained reference/firm bar configuration |
| Partial / Multi-cut | Experimental; distinct-depth ligament and one four-slice pattern pass |
| TRIM | Unsupported by the CUT profile; v7 and finite v7_1 primary failures retained |
| Reverse / oblique | New adoption NOT_RUN; earlier slotted-board evidence retained |

The original held-out task denominator is **2/4 per solver**, although all four
cases truly sever into two parts. The ±6 mm cases fail the unchanged 2 mm
reference-midpoint gate (even ideal geometric labels score 2.5 mm). The wider
proposal is rejected. Final use is restricted after those failures to +x blade
direction, ±5° yaw, -4/0/+4 mm cut positions and 8 mm bar-length increments,
within the declared 1 mm particle / 4 mm grid / 0.1 ms configuration. These
finite results do not validate all combinations or a random success rate.
TRIM failure does not veto proven CUT; surface appearance does not certify it.

```python
from robotic_cutting_sim.scene_config import compile_task

task = compile_task("configs/wp12/scenes/bar_flat_board.json", "sever_mid",
                    solver_profile="v7_cut_research_v1")
# Explicit original v7, recorded path_v2 controller, 14 s completion window.
```

[Normal Scene / Goal / Environment selection](docs/V7_RESEARCH_WORKFLOW.md) is supported.
Configuration selection checks and records the solver/front/contact sources,
material, scene and numerical scope; incompatible overrides raise. New policy
actions require separate evaluation. `tlmpm_v7_1` stays Experimental and is not
selected by this profile. Its post-study pending-event lifecycle correction is CPU-verified only;
GPU outcomes are not reassigned to it. Materials are uncalibrated; forces are not converged.
[Results, all failures and exact commands](docs/reports/WP14_RESULTS.md) ·
[Profile configuration](configs/wp14/v7_cut_research_v1.json).

## Research limits and retained history

Original v7 remains opt-in; full/default promotion was rejected. The primary
kinematic TRIM stays connected and a held-out shorter/firmer CUT fails.
The later scoped CUT qualification does not rewrite those failures or establish
general TRIM support. Forces remain unconverged and materials uncalibrated.
[Frozen WP-13 results](docs/reports/WP13_RESULTS.md) ·
[Historical media and baseline evidence](docs/HISTORICAL_CAMERA_MEDIA.md).

Both surface profiles are one-way render geometry, separate from collision.
Unchanged `particle_surface_v1` and opt-in `cut_aware_surface_v1` were measured on
saved states; no plane smoothing, particle removal or visual tear splitting was
adopted. The historical WP-12 cut-aware surface gate remains FAIL.
[Original surface decision](docs/reports/WP12_PHASE3_XHIGH_DECISION.md).

## Quick start

Use Python 3.12 and the pinned, separate base/Native environments.
[Setup and hardware requirements](docs/SETUP.md) · [compatibility](docs/COMPATIBILITY.md).
The bootstrap installs project-local dependencies; it does not upgrade drivers.

```bash
bash scripts/bootstrap.sh --runtime base --managed-python
bash scripts/bootstrap.sh --runtime isaac_sim --managed-python --accept-eula
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-/tmp}"

# CPU-only: compile the exact WP-12 straight-cut scene and live goal.
.venv/bin/python scripts/run_wp12_workcell_live.py prepare \
  --episode CUT --out artifacts/wp12/my_cut_inputs

# Native episode, using the existing physical-grasp runner (default GPU 0).
# Choose fresh output paths and run only when that GPU is available.
.venv/bin/python scripts/run_wp12_workcell_live.py launch \
  --episode CUT --out artifacts/wp12/my_native_cut
```

The Native runner checks the motion plan before launching, records RGB-D and
oracle data separately, and reports task and HOME outcomes separately. Output
directories are never overwritten. The prepare command is a configuration check;
it does not establish cutting success.

The same scene/goal configuration is available through the Python API:

```python
from robotic_cutting_sim.scene_config import compile_task
from robotic_cutting_sim.live_env import open_cutting_env

task = compile_task("configs/wp12/scenes/bar_flat_board.json", "sever_mid", seed=42,
                    solver_profile="v7_cut_research_v1")
# CPU-only configuration inspection:
print(task.task_id, task.solver_module, task.max_cuts)

# For interactive control, open the task in Native Isaac Sim:
# with open_cutting_env(runtime="isaac_sim", task=task,
#                       preset="cutting_research_v1",
#                       output_dir="artifacts/wp12/my_api_run", seed=42) as env:
#     obs, info = env.reset(seed=42)
#     obs, reward, terminated, truncated, info = env.step(action)
```

[Environment API and actions](docs/POLICY_READY_API.md) ·
[scene/goal/controller separation](docs/adr/0015-scene-goal-controller-evaluation-separation.md) ·
[task switching and observation isolation](docs/reports/WP12_AGENTIC_READINESS.md).
Deployable observations exclude contact force, particles and material cut codes.
The kinematic runtime has ideal tracking and no cameras; it is a physics diagnostic.

## Supported scope and retained WP-12 limits

| Capability | Evidence / boundary |
| --- | --- |
| Solid-board vertical sever, TRIM, multiple slices and mixed operations | [WP-12 Native Demos 1–6](docs/reports/WP12_FINAL_VALIDATION.md); open-loop bar sliding can move later cuts off their declared planes |
| Oblique cutting | The WP-12 20° Native demo uses an explicit slotted board; arbitrary solid-board tilt is unsupported |
| Generalization | 170 saved kinematic runs; feasible ID_validated **47/61 = 77%**, Wilson 95% [65%, 86%]; original ≥90% target **NOT MET** |
| Multi-slice family D | 0/20 benchmark successes; later cut locations fail after material slides; frozen failures retained |
| Surface quality | PARTIAL; cut-aware gate FAIL, trim spikes and thin-chip tears remain |
| Contact forces / materials | Force **NOT CONVERGED** with timestep/resolution; material uncalibrated |
| Repeatability / memory | GPU debris counts vary; Demo 5 A/B has 5 vs 9 components, cause OPEN; kinematic host-memory gate FAIL |

[Generalization and Native sample coverage](docs/reports/WP12_GENERALIZATION.md) ·
[physics envelope](docs/reports/WP12_CUTTING_PHYSICS.md) ·
[resolution study](docs/reports/WP12_RESOLUTION_STUDY.md).

## Documentation and tests

- [Documentation index and repository map](docs/README.md)
- [WP-12 final validation and acceptance matrix](docs/reports/WP12_FINAL_VALIDATION.md)
- [Current progress and preserved failures](docs/PROGRESS.md)
- [WP-11 history](docs/WP11_PLAN.md) · [frozen BladeMaster demonstrations](docs/reports/BLADEMASTER_DEMOS.md)
- [Contributing](CONTRIBUTING.md) · [asset sources](assets/ASSET_SOURCES.md)

```bash
# CPU tests; genuine local-artifact checks may explicitly skip on a fresh checkout.
.venv/bin/python -m pytest -q -rsx
```

GPU probes run separately. A skipped check is NOT_RUN; successful execution is
separate from task success, physical fidelity and real-material calibration.

## RL-ready (Phase 1)

`open_vector_env` runs N cutting envs in **one process with batched GPU kernels**: one batched `tlmpm_v7_2`
material and one CUDA graph per batch shape. It never loops over single envs or starts N Kit processes; an
unsupported configuration raises an `UNSUPPORTED_*` code. Plan and status are in
[RL_READY_PLAN.md](docs/RL_READY_PLAN.md) and [RL_READY_PROGRESS.md](docs/RL_READY_PROGRESS.md). The final
feature-parity matrix and Phase 1 checklist are in
[P1_PARITY_MATRIX.md](docs/reports/rl_ready/P1_PARITY_MATRIX.md); the whole plan is summarised in the
[final report](docs/reports/rl_ready/FINAL_REPORT.md) (draft). Phase 1 reached `main` in PRs #12–#15.

**Backends of `open_vector_env(num_envs, task=..., tasks=..., device="cuda", backend=...)`:**

- `kinematic_ideal_tracking_batched` (default, workloads A / C). The edge is replayed or ideally tracked, with no
  robot.
  - Tasks: seeded TRIM (`TaskFamily`, `tasks=[512, 15]`) or configured `tlmpm_v7_2` scenes
    (`ConfiguredTaskFamily`, e.g. the CUT workload `cut_v7_2_v1`, or `tasks=[{"scene": "cut_v7_2_v1", "seed": 3}]`).
  - Fidelity: always `reduced_fidelity`.
- `isaac_lab_newton` (workload B). N Franka worlds on Isaac Lab Newton / MuJoCo-Warp, coupled to the same batched
  material.
  - `tool_mount="rigid_mount_v1"` (B1): `reduced_fidelity: rigid_tool_mount_no_grasp`.
  - `tool_mount="physical_grasp_v1"` (B2) with the opt-in `grip_contact_profile="mjwarp_grip_v1"` (a listed
    difference, not a calibrated grip).
  - Needs `.deps/isaaclab-venv`.
- `kinematic_grip_batched`: the Phase 2 rubber strip task (see [Rubber material (Phase 2)](#rubber-material-phase-2)).

**Particle-contact modes** (option `particle_contact_mode=` of the default backend; all three compute the same
contacts and impulses on the same state):

- `frozen_v1` (default): the frozen contact path, byte-identical.
- `alias_free_v1` (opt-in, P1-3a): alias-free hash dimensions and a speed-guarded fallback. In the exclusive table it
  is 1.38–1.47× faster than `frozen_v1` in env-steps/s, but the workload-A windows at N ≥ 4 are pre-contact.
- `alias_free_v2` (opt-in, P1-6 cycle `p16c1`, PR #14): env-major hash buckets plus a per-env query shrink while
  every particle of that env is slower than 0.25 m/s. It was written by the external agentic optimizer and accepted
  by the frozen evaluator at 1.168× (N=1) and 1.279× (N=16) over `alias_free_v1` (`SHARED_GPU`, interleaved ABBA
  pairs). Limits:
  - the shrink engages only in slow envs; `particle_contact_options={"shrink_speed_m_s": None}` turns it off;
  - O(N·C) bucket memory (about 4.3 MiB/env measured) and an int32 cap of N ≤ 2495 for the TRIM workpiece;
  - a diagnostic per-env counter `shrink_engaged_substeps` (PR #15, receipt only).

**Both cutting backends share one contract (`rl.vector_contract`):**

- `reset(seeds=, env_ids=)` (masked partial reset); `step(actions[N,A])` returns Torch device tensors;
- `info["termination_reason"]` (`rl.termination`: material, rigid, grasp lost, IK, truncation, input guard,
  infrastructure; `decode_termination_detail` names the material bits);
- `evaluate(env_ids)` through the **frozen** `goals.evaluate_goal`. A non-finite env gets a structured
  `FAIL_INTEGRITY` entry;
- `receipt()` with computed fidelity.

The rubber backend `kinematic_grip_batched` has its own contract (`rl.rubber_contract`, `rubber_vector_contract_v1`,
actions `[N, 6]`) and its own evaluator (`rubber_tracking_eval_v1`, not `goals.evaluate_goal`). It shares only the
`reset` / `step` / `termination_reason` / `privileged_info_keys` conventions.

**Around the env:**

- the P1-4 supervisor and crash-safe ledger (`rl.supervision`, `scripts/rl_ready/supervise_rollout.py`);
- the RL pipeline smoke (`scripts/rl_ready/run_rl_smoke.py`, `rcs_rl_rollout_v1`; opt-in `--rollout-schema
  rcs_rl_rollout_v2` labels the privileged / oracle-derived fields; no training);
- the P1-6 acceptance harness and judge (`scripts/rl_ready/run_perf_acceptance.py`, `judge_perf_acceptance.py`) with
  the **frozen** contract `configs/rl_benchmark/acceptance_v2.json` (D-14, frozen at the PR #13 merge `bfe6d5e`).
  The judge recomputes PHYSICS, PERFORMANCE and INTEGRITY layers from the candidate's dumps. G9 times steps 229–400
  of the s512 replay (all tool contact and cutting) at N 1/16 with 3 ABBA pairs. `acceptance_v1.json` (pre-contact
  window) is frozen and kept for the record only; any change is `acceptance_v3`.
- the external agentic optimizer cycle (P1-6). The optimizer works in its own repository and edits an isolated RCS
  worktree. A candidate counts only if the frozen evaluator accepts it, an RCS PR merges it, and a new versioned
  descriptor pins the merged commit. Cycle `p16c0` pinned `rcs_rl_vector_v1` at `bfe6d5e`. Cycle `p16c1` produced
  `alias_free_v2`: accepted, merged at `9ee1e9d`, pinned as `rcs_rl_vector_v2`, **SYNCED**. Two earlier acceptance
  attempts were BLOCKED by evaluation-sandbox bugs, and one was stopped because it starved on the GPU lock; none of
  them was counted.

```bash
# CPU tests (project venv); run once with CUDA hidden and once with GPU 0 visible.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/
# Torch vector-env / RL tests on the Warp/Torch CPU device (D-09 pytest overlay, nothing installed into .deps)
PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_rl_*.py tests/test_vector_env_*.py tests/test_perf_acceptance.py
# Isaac Lab workload B tests (Warp-CPU MJWarp)
PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaaclab-venv/bin/python -m pytest -q \
  -p no:cacheprovider tests/test_rl_lab_*.py tests/test_vector_env_contract.py
# GPU smokes: one RCS job per GPU under its flock, explicit device, >= 4 GB left free
CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock \
  .deps/isaacsim-venv/bin/python scripts/rl_ready/run_p1_2b_smoke.py --backend kinematic_ideal_tracking_batched \
  --gpu-index 0 --out artifacts/rl_ready/<new_run>
CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock \
  .deps/isaacsim-venv/bin/python scripts/rl_ready/run_rl_smoke.py --run S-CUT --gpu-index 0
```

**Known limitations:**

- Workload A and B1 are `reduced_fidelity`, and every receipt says why. Batched CUT is the versioned v7_2 scene
  only; frozen v5 / v7 CUT stays single-env / Native (`UNSUPPORTED_SOLVER`). B2 grip slip is about 1.5 mm,
  against about 1 µm in Native.
- Cameras, vision profiles and physics recorders are **UNSUPPORTED** in the batched path (`UNSUPPORTED_SENSOR` /
  `UNSUPPORTED_RECORDER`). Vision data comes from the Native reference.
- Warp 1.17 replays a captured HashGrid query with the descriptor of the last-built grid. So the rule is **one
  device HashGrid per process**: one batch or single material per process (`UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS`;
  a grid created later sets `FOREIGN_DEVICE_HASHGRID`, which counts as material-invalid plus infrastructure). An
  early "≈1.9× at N=4" timing came from a process holding two grids and is withdrawn.
- Performance: the formal table is merged —
  [P1 performance table](docs/reports/rl_ready/P1_PERFORMANCE_TABLE.md) (GPU1, RTX 5090, 20 of 23 runs
  `EXCLUSIVE_GPU`; particles/env 102,732 (A, s512), 96,000 (CUT) or about 103,300 (B, seeds 512/15 mixed); 50 Hz
  control, material dt 0.1 ms = 200 substeps; headless rows have no cameras and no recording in the timed calls;
  warm). Headline env-steps/s:
  - Native D-s512 (full RGB-D rig): 4.04; frozen single-env headless replay s512: 7.57;
  - workload A `frozen_v1`: 6.75 at N=1, 9.83 at N=16; opt-in `alias_free_v1`: 9.74 at N=1, 13.56 at N=16
    (A windows are pre-contact at N ≥ 4);
  - batched CUT (v7_2 scene): 6.85 at N=1, 10.35 at N=8, 10.22 at N=16;
  - workload B1 (rigid mount): 4.96 at N=1, 7.66 at N=4, 8.67 at N=8; N=16 (8.79) is `SHARED_GPU` only; B2 N=4: 7.37.

  The rows measured code `691b571`, before the per-step HashGrid refresh of review fix F1. P1-4 long-rollout rows
  and every partial-reset latency are `SHARED_GPU`. `alias_free_v2` has no row in that table; its accepted ratios
  are `SHARED_GPU`. No B-vs-Native ratio is claimed.
- "Whole-RCS RL-ready" is **PARTIAL for CUT**: the frozen v5 CUT is not batched, and batched CUT is the opt-in
  v7_2 scene with ideal tracking and no robot.
- Materials are provisional and uncalibrated. The rubber material is described below.

Reports in `docs/reports/rl_ready/`:
[P1-0 baseline](docs/reports/rl_ready/P1_0_BASELINE.md) ·
[P1-1/P1-2 milestones](docs/reports/rl_ready/P1_1_P1_2_MILESTONES.md) ·
[P1-1b robot batching](docs/reports/rl_ready/P1_1B_ROBOT_BATCHING.md) ·
[P1-3a contacts](docs/reports/rl_ready/P1_3A_ALIAS_FREE_CONTACTS.md) ·
[P1-4 long rollout](docs/reports/rl_ready/P1_4_LONG_ROLLOUT.md) ·
[P1-5 CUT](docs/reports/rl_ready/P1_5_CUT.md) ·
[P1-6 acceptance harness](docs/reports/rl_ready/P1_6_R0_ACCEPTANCE_HARNESS.md) ·
[P1 parity matrix](docs/reports/rl_ready/P1_PARITY_MATRIX.md) ·
[P1 performance table](docs/reports/rl_ready/P1_PERFORMANCE_TABLE.md) ·
[P1-1/P1-2/P1-3a acceptance review](docs/reports/rl_ready/P1_1_P1_2_P1_3A_ACCEPTANCE_REVIEW.md) ·
[Phase 1 acceptance review](docs/reports/rl_ready/PHASE1_ACCEPTANCE_REVIEW.md) (PASS_WITH_LIMITATIONS at `f03fbff`) ·
[P1 follow-up](docs/reports/rl_ready/P1_FOLLOWUP.md) (`acceptance_v2`, review findings 1–4, 10, 12) ·
[P1 follow-up review](docs/reports/rl_ready/P1_FOLLOWUP_REVIEW.md) ·
[P1-6 cycle p16c1](docs/reports/rl_ready/P1_6_p16c1.md) ·
[PR #14 review](docs/reports/rl_ready/P1_6_P16C1_PR14_REVIEW.md) ·
[p16c1 follow-ups](docs/reports/rl_ready/P1_6_P16C1_FOLLOWUPS.md) ·
[PR #15 review](docs/reports/rl_ready/P1_6_P16C1_PR15_REVIEW.md).

## Rubber material (Phase 2)

Phase 2 adds a rubber material to the same batched material layout and vector-env contract as Phase 1. It reaches
`main` in its own PR. Material status everywhere: **`SIMULATOR_ONLY_UNCALIBRATED`**. No real rubber data exists, and
every modulus, Prony term and friction value is illustrative. Design: [ADR 0023](docs/adr/0023-rubber-material-v1.md);
summary and every recorded FAIL: [final report](docs/reports/rl_ready/FINAL_REPORT.md).

**What exists (all opt-in; frozen solvers v1–v7_2 and Phase 1 modules are unchanged):**

- **Solver `tlmpm_rubber_v1`**, as a single-env reference (`LiveRubberMaterial`) and batched
  (`batched/rubber_material_batch.py`, `BatchedRubberMaterial`). Two models:
  - `neo_hookean_jbar_v1`: isochoric Neo-Hookean with a J̄-averaged volumetric term and a documented reduced bulk
    ratio κ = 20 μ;
  - `neo_hookean_jbar_prony_v1`: the same plus Prony viscoelasticity (exact exponential update). The internal
    variables are per env and are cleared by a partial reset.

  Recipes: `configs/materials/rubber_generic_{soft,visco}_v1.json`.
- **Contact laws `grid_node_penalty_coulomb_v1/v2/v3`.** v1 is the code default and has the worst formal record
  (see the limitations below). v2 is the law of the RL task. v3
  (patch slip-rate equalisation) is valid for straight translational sliding only: review finding K2-1 showed an
  unbounded patch torque under rotation or sideways drag.
- **S1 self-penetration detector** (`surface_grid_v1`): detection only. It sets the sticky invalid bit
  `RUBBER_SELF_CONTACT_UNRESOLVED`. Self-contact forces are not implemented.
- **Batched rubber RL task:** `open_vector_env(backend="kinematic_grip_batched")` with a `RubberStripTaskFamily`.
  - Task and profile: task `rubber_strip_drag_v2`, profile `rubber_rl_env_v2`.
  - Scene: a strip on a table, dragged by a kinematically tracked grip toward a tip goal. Slots are L 100 / 125 /
    150 × t 4 / 5 / 6 mm, W 16 mm, dx 1 mm.
  - Actions and observations: 6-D `grip_delta_v1` actions; a 78-dim deployable `policy` observation; an opt-in
    106-dim `critic` observation with `observation_profile="privileged_state"`.
  - Reward and success: the shaping reward `rubber_tip_tracking_v1`; success comes only from `evaluate()`
    (`rubber_tracking_eval_v1`).
  - Randomisation: material, traction, preload and goal are resampled per reset. Geometry is fixed per env;
    `rebuild()` changes it.
- **Native rubber host** (`native_rubber_host.py`, launcher `scripts/rubber/p2_6/launch_native_rubber.py`). It runs
  the same recipe and profile inside Kit, fed by the PhysX-measured clamp pose.
  - RL-09 PASS: the material inputs were bitwise equal in 30 headless replays, and 0 of 46 statistics fell outside
    the declared band.
  - The clamp wrench is reported only, not fed back.
- **Calibration interface for real data:**
  - `rubber_calibration_v2.py`: manifest schema `rcs_rubber_calibration_dataset_v2` (data origin, sensors, measured
    specimens) and `real_calibrated_gate_v2`, which refuses synthetic and v1 manifests;
  - batched forward models of five experiment types (`rubber_calibration_experiments.py`);
  - a fit API (`rubber_calibration_fit.py`: bounded TRF, CMA-ES, an ask/tell plug-in point) with a one-shot held-out
    ledger.

  The real-data manifest `configs/rubber/calibration/none_v1.json` is `NO_REAL_DATA`, and the real-data protocol
  `rubber_calib_tests_real_v1` exists only as an outline (no thresholds; NOT_RUN). Read the guidance for real data in
  [P2_5_RUBBER_CALIBRATION.md](docs/reports/rl_ready/P2_5_RUBBER_CALIBRATION.md) §12.2 first.

  Calibration results do not plug directly into the RL task. P2-5 fits friction with contact law v3, while the RL
  task uses law v2 at μ 0.3. Fitted θ are specific to n_t 4, while the RL slots run at n_t 4 / 5 / 6. The RL task
  draws its material from the ranges of `rubber_strip_drag_v2`, not from a recipe. A real-data user needs a new task
  config, a law-v2 friction fit and the resolution bias shift (final report §5).

```python
import torch
from robotic_cutting_sim.rl.vector_env import open_vector_env
from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily

N = 8
family = RubberStripTaskFamily(num_envs=N, slot_seed=0)        # rubber_strip_drag_v2, profile rubber_rl_env_v2
with open_vector_env(num_envs=N, task=family, backend="kinematic_grip_batched", device="cuda") as env:
    obs, info = env.reset(seeds=[1000 + e for e in range(N)])  # obs["policy"]: float32 [N, 78] on the device
    actions = torch.zeros((N, 6), dtype=torch.float32, device=obs["policy"].device)
    obs, reward, terminated, truncated, info = env.step(actions)
    obs, info = env.reset(env_ids=[3], seeds=[2000])           # masked partial reset with new draws
    results = env.evaluate()                                   # rubber_tracking_eval_v1
```

Set `OPENBLAS_NUM_THREADS=1` (or a small value) in any process that opens the rubber env. A reset recomputes block
inverses on the host, and with OpenBLAS's default all-core pool on a loaded host the reset p95 was 5.0 s, against
0.093 s single-threaded. The inverse's bits depend on the thread count.

**Known limitations:**

- Table friction is **fixed at μ = 0.3**. The law-v2 envelope check failed at 0.4 on the strip scene; real rubber is
  often 0.6–1.0 or more.
- Robot-coupled rubber is deferred: the grip is kinematic, and no rubber wrench reaches a robot. The 20 ms handoff
  was assessed as significant.
- Throughput at full fidelity (n_t 4, 5,226 substeps per 20 ms step) is about **1 env-step/s** aggregate: 0.96 at
  N=8 for the full slot set, `SHARED_GPU`. The labelled reduced-fidelity variant `rubber_rl_v1` (n_t 3) is about
  3× faster.
- Bending carries a measured discretisation bias at n_t 4: +0.62 % (L/t 20) and +0.71 % (L/t 37.5). The
  RUBBER-PHYS-11 v5 FAIL is kept.
- **P2-5 synthetic calibration test: FAIL** (`rubber_calib_tests_v2`). The interface, batched-equals-single parity,
  sealing and one-shot machinery work. The declared staged fit stopped on its per-stage evaluation caps and failed fit
  quality, held-out prediction and parameter recovery. A final blind re-test was estimated at 7.1–10.2 GPU-h and was
  not run.
- The near-incompressibility treatment (reduced bulk ratio κ = 20 μ) is a model-form error: volume ratio J − 1 =
  +5.4 % and nominal stress −4.2 % at stretch 2.0, and bending stiffness −1.6 % (ADR 0023 §6). Do not read volume
  change or confined responses as rubber physics.
- The code default contact law is v1, which failed RUBBER-CONT-01 / 03 / 04 / 05. The friction envelope is checked only
  in the RL task family. Both recipes carry a nominal table friction of 0.8, so a material built directly from a
  recipe without an explicit law runs v1 at μ 0.8 with no refusal.
- A fitted θ is **resolution-specific** (`rubber_ref_v1`, n_t 4). Do not carry it to another resolution without the
  bias shift.
- No cameras, sensors or recorders in the batched rubber env (`UNSUPPORTED_*`). The P2-6 env tests need Torch, so
  they are NOT_RUN in GitHub CI; the local Torch-venv runs are the gate.

```bash
# Rubber CPU suites (project venv; Torch-gated tests skip there = NOT_RUN)
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rubber_*.py \
  tests/test_live_rubber_material.py tests/test_tlmpm_rubber_v1.py tests/test_native_rubber_*.py tests/test_repo_naming.py
# Rubber RL env tests (Torch venv, D-09 pytest overlay)
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src \
  .deps/isaacsim-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_rubber_rl_*.py \
  tests/test_rubber_device_staging.py tests/test_rubber_scripted.py tests/test_native_rubber_*.py
# GPU rows (P2-6 runner) under the blocking per-GPU flock; sets OPENBLAS_NUM_THREADS=1
# Row names: RL01 RL02b RL02c RL03 RL04b RL05 RL06 RL07 RL08 RL08b
scripts/rubber/p2_6/queue_gpu_rows.sh <gpu index> <rows, e.g. RL01,RL02b> artifacts/rl_ready/<new_run>
```

Reports in `docs/reports/rl_ready/`:
[P2-0/P2-1 rubber](docs/reports/rl_ready/P2_0_P2_1_RUBBER.md) ·
[P2-2 hyperelastic](docs/reports/rl_ready/P2_2_RUBBER_HYPERELASTIC.md) ·
[P2-3 viscoelastic](docs/reports/rl_ready/P2_3_RUBBER_VISCO.md) ·
[P2-4 contact](docs/reports/rl_ready/P2_4_RUBBER_CONTACT.md) ·
[P2-5 calibration](docs/reports/rl_ready/P2_5_RUBBER_CALIBRATION.md) ·
[P2-6 rubber RL](docs/reports/rl_ready/P2_6_RUBBER_RL.md) ·
reviews: [P2-0..P2-2](docs/reports/rl_ready/P2_0_P2_2_ACCEPTANCE_REVIEW.md),
[P2-3](docs/reports/rl_ready/P2_PREREQ_P2_3_REVIEW.md), [K2 combined](docs/reports/rl_ready/P2_K2_COMBINED_REVIEW.md),
[P2-5](docs/reports/rl_ready/P2_5_ACCEPTANCE_REVIEW.md), [P2-6](docs/reports/rl_ready/P2_6_ACCEPTANCE_REVIEW.md).

## Agentic control robot seeded trimming

The opt-in [seeded trimming challenge](docs/AGENTIC_TRIM_CHALLENGE.md) provides
actual live rig14 Global RGB and wrist_0 RGB-D, explicit arm_0 actions and
independent physical scoring. The [executed bounded study](docs/reports/AGENTIC_TRIM_STUDY.md)
compared two agent-authored methods on real seeded scenes. Runtime failures
prevented mechanical performance measurement; no autonomous cutting success
was established. All original physics and historical evidence remain preserved.

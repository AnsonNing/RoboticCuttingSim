# RL-Ready Simulator & Rubber Physics — work plan, acceptance and decisions

Source of requirements: [RL_READY_SPECIFICATION.md](RL_READY_SPECIFICATION.md)
(faithful copy of the user's plan v2.0 and brief v2, 2026-11-10). This file turns
it into work packages with acceptance criteria, fixes the measurement boundaries
and records decisions. Progress: [RL_READY_PROGRESS.md](RL_READY_PROGRESS.md).

Status words for every check: **PASS / FAIL / BLOCKED / NOT_RUN** (a skipped test
is not a pass; exit code 0 is not task success). Cross-repository sync status:
REJECTED_PHYSICS / REJECTED_PERFORMANCE / RCS_PR_PENDING /
MERGED_RCS_PENDING_OPTIMIZER_SYNC / SYNCED (decision D-02).

## 1. Fixed rules (from the specification and AGENTS.md)

- RCS is the simulator source of truth; no separate RL simulator, no hidden copy.
- **True batching only**: one process, batched GPU kernels over N envs. A Python
  loop over single envs, or N Kit processes, is never called vectorized; an
  unsupported configuration raises `UnsupportedVectorConfiguration` (message starts
  with `UNSUPPORTED`).
- Physics fidelity is never lowered silently. Any reduced workload carries
  `reduced_fidelity: true` and is reported in separate rows.
- Frozen and untouched: solvers v1–v7_2 and their defaults, law 8, v1 precision
  benchmark/`evaluate_oracle`, final_v4, camera rig v1–v5 evidence, evaluation v2
  layers, WP-11..17 records. New behaviour is versioned and opt-in.
- WP-17 seed 512 final cut-code difference vs WP-16 remains an **open issue**;
  no physics-equivalence claim without a measurement.
- Deployable observations never contain oracle material state, cut codes or
  contact truth. Materials stay provisional/uncalibrated.

## 2. Runtimes and measurement boundaries (decision D-03)

| Boundary | What runs | Includes | Excludes |
| --- | --- | --- | --- |
| `single_env_native` | `open_cutting_env(runtime="isaac_sim")` (Native Isaac Sim 6.2, PhysX Frankas, physical grasp, RTX cameras) — the `FULL_NATIVE_REFERENCE` profile | factory/reset, every action, evaluation, media as declared per row | nothing hidden; startup reported separately from warm stepping |
| `single_env_headless` | the existing single-env material (`LiveCuttingMaterial`, one CUDA graph of 200 substeps) driven without Kit — reference wrapper | material graph, coupling bookkeeping, state observation | cameras, Kit, recording (unless the row says so) |
| `batched_headless` | the new `rl_gpu_headless` runtime: N envs in one process, batched kernels | batched material + coupling (+ batched robot when the row says so), Torch obs/reward/done | per-env Python loops (forbidden) |

Vision vs state is a separate column (`cameras`); a state row never initialises
RTX cameras. Every performance row reports:

`N | particles/env | control Hz | material dt | cameras | env-steps/s | aggregate sim-time/wall | VRAM peak | reset p95 | physics verdict`

plus substeps/control step, robot DOF, contacts enabled, recording, warm/cold,
exact GPU (name, index, driver), `SHARED_GPU` or `EXCLUSIVE_GPU` with sampled
co-tenant utilisation, source commit and workload id. Timings of different
boundaries are never merged into one speedup number.

## 3. Benchmark workloads (`configs/rl_benchmark/`)

| Id | Workload | Physics | Phase |
| --- | --- | --- | --- |
| A | physics-only material: representative TRIM workpiece (seeded_trim, tlmpm_v7_2, law 8, 1 mm / 4 mm / 0.1 ms, 200 substeps per 50 Hz step) driven by a recorded measured tool trajectory, real cutting enabled | full | P1-0 |
| B | robot + material + contact: Franka(s) + tool + material coupling with delayed reaction wrench | full | P1-1 (robot batching) |
| C | original CUT (CUT-01 family / configured cut scenes) | full | P1-5 |
| D | original TRIM (seeded_trim `rcs_trim_random_sim_v1`, seeds 512 / 15, WP-17 configuration) | full | P1-0 replay, P1-5 live |
| E | dynamic rubber (`rubber_material_v1`) | full | Phase 2 |

Frozen regression inputs (decision D-05): the complete WP-17 Native episodes
(seed 512 and seed 15, `artifacts/wp17/native_optimized_001/`) contain 1200
submitted actions, measured body poses of all 23 bodies at every control step and
121 streamed physical samples (positions, velocities, cut codes). Replaying the
measured knife/finger trajectory through the material reproduces the Native
material input exactly up to interpolation; this is the headless↔Native and
single↔batched material parity reference. Replay results are compared with the
Native samples, never substituted for them.

## 4. Work packages and acceptance

### P1-0 — baseline and benchmark runner (first)

Deliverables: environment inventory (git HEAD, Python/Warp/Newton/Isaac Sim/
Isaac Lab/Torch, GPU, driver, memory); `configs/rl_benchmark/` workloads;
benchmark runner with the three boundaries, GPU telemetry sampler (co-tenant
utilisation, VRAM), JSON + Markdown output; stage/kernel breakdown via the
existing `profiling.py` (no new profiler); single-env headless replay of the
WP-17 trajectories with comparison to the Native samples.

Acceptance: inventory and baseline JSON committed under `docs/evidence/rl_ready/`;
single_env_headless workload A/D rows measured (warm and cold separately); kernel
stage split reported; `single_env_native` row taken from WP-17 evidence (labelled
as such) — a new Native run only if needed; replay parity numbers (final codes,
labelled-count difference, max position difference over time) reported.

### P1-1 — true GPU batched physics

Deliverables: batched tlmpm_v7_2 material with per-env isolated particles,
velocities, F, material parameters, grid/slots, cut codes/history/pending
frontier, tool poses/twists, cut index, stroke, reaction/obstacle wrenches,
statistics, clock and seed; one CUDA graph per batch shape; batched coupling
(GPU pose interpolation, one `[N,6]` reaction readback); robot batching route per
ADR 0021. Existing single-env API remains the reference wrapper.

Acceptance:
- CPU (Warp CPU device) parity: batched N=1 and every env of N identical copies
  equal the single solver (bitwise where the ADR proves it achievable, otherwise
  declared tolerance) over a short cutting replay.
- Cross-env isolation: perturbing one env's input leaves the others unchanged.
- GPU: N=1/4/8/16 batched replay of workload A runs; per-env invariants (mass,
  positive detF, code monotonicity, no NaN) PASS; single↔batched GPU differences
  reported with tolerance.
- Profiler shows batched kernels (one launch over all envs), not N launches.

### P1-1b — robot batching (workload B)

After M2 of workload A, under a separate ADR (ADR 0021 §9): Isaac Lab DirectRLEnv on Newton/MuJoCo-Warp with N
worlds (`{ENV_REGEX_NS}` scene, per-world body indices, batched IK, graph-safe per-world wrench), first B1 rigid
knife mount (`reduced_fidelity: rigid_tool_mount_no_grasp`), then B2 physical grasp. Acceptance: N=1/4/16 real
batched articulation + batched material coupling with the delayed wrench, per-env wrench isolation test, Native
confirmation of material parity on identical edge sequences.

### P1-2 — vectorized environment API + fast/partial reset

Deliverables: `open_vector_env(num_envs, runtime="rl_gpu_headless", task, material,
observation_profile="state", device="cuda")`; `reset(seeds=)`, `reset(env_ids=)`,
`step(action[N,A])` → obs, reward, terminated, truncated, info as Torch tensors on
the device; per-env time/seed/RNG/goal; deployable vs privileged split preserved;
Gymnasium single-env adapter retained; Isaac Lab-compatible batched runner.

Acceptance (M0/M1/M2):
- M0: N=1 complete step/reset; parity with the single reference within the
  declared range.
- M1: N=4 on one GPU; env independence and partial reset PASS (non-reset envs
  bit-identical to an un-reset run on the same device).
- M2: N=8–16 continuous rollout with measured throughput gain over N=1.
- API tests: shapes, dtypes, device, no CPU round trip per env, deployable
  observation contains no oracle fields.

### P1-3 — headless runtime, kernel and data-movement optimisation

Only after M2. Profile-driven; CUDA graph replay kept; fewer host syncs and
uploads; candidates (hash, contacts, frontier, reductions) need hotspot evidence.
Acceptance: stage breakdown per runtime, measured improvement, unchanged physics
invariants and replay parity, rollback switch.

### P1-4 — memory, stability, long rollout

Acceptance: long vectorized rollout with repeated partial resets; bounded VRAM and
RSS (slope reported); 8+ serial full episodes; infrastructure failures (OOM,
kernel error, crash) reported apart from task failures; known Native
`reload_in_place` / camera-off limitations retained.

### P1-5 — whole-RCS feature parity

Feature-parity matrix (scene/goal/material schema, CUT, TRIM, robot articulation/
contact, cut history, cameras, vectorized reset/step, GPU-native tensors, seeded
randomisation, profiling, rubber placeholder) with RL_GPU_HEADLESS vs
FULL_NATIVE_REFERENCE status. Representative CUT and TRIM actually run in the
batched path with the independent physical evaluator; gaps stay PARTIAL.
RL smoke: random/scripted policy over N envs with repeated partial resets,
saved obs/action/reward/done consumable by a standard RL loop (no training).

### P1-6 — external agentic optimizer improves RCS performance

Isolated RCS candidate worktree → CPU + GPU + physics regression → frozen
candidate commit → independent acceptance → RCS PR → merge → new versioned
descriptor pinned to the merged SHA in the external repository (frozen checkout,
historical descriptors untouched) → smoke → SYNCED, with a sync receipt.
Evaluators are never modified by the optimizing agent.

### Phase 1 done

All seven checkboxes of the specification's "Phase 1 Done" with evidence; xhigh
acceptance review PASS (or PASS_WITH_LIMITATIONS written into PR and docs);
merged to `main` through a PR; external descriptor pinned (SYNCED).

### Phase 2 — rubber material (after Phase 1 merge)

P2-0 recipe + calibration manifest interface (status
`SIMULATOR_ONLY_UNCALIBRATED`); P2-1 no-tool deformation tests (RUBBER-PHYS-01..);
P2-2 hyperelastic (compressible Neo-Hookean baseline, near-incompressibility
handled explicitly, J = det F tracked, refinement study); P2-3 viscoelastic
(Prony/Maxwell internal variables, rate dependence, relaxation/hysteresis,
dt-consistent time scales, reset per env); P2-4 contact/friction (table/gripper,
slip onset, penetration, 20 ms coupling-delay assessment, per-env wrench
isolation); P2-5 calibration interface + held-out protocol (frozen before held-out;
numerical validation only without real data); P2-6 RL-ready rubber acceptance
(N=1 and ≥8 envs, randomised material/geometry/friction, Full Native usability,
throughput/VRAM vs Phase 1). Done per the specification's "Phase 2 Done".

## 5. Decisions

| Id | Date | Decision | Reason |
| --- | --- | --- | --- |
| D-01 | 2026-11-10 | The verbatim user documents stay outside the repository; `RL_READY_SPECIFICATION.md` is the in-repo copy with the naming substitution only. | Repository naming rule. |
| D-02 | 2026-11-10 | The plan §7 status word naming the external project is written `MERGED_RCS_PENDING_OPTIMIZER_SYNC` in RCS; the external repository uses the original word. | Naming rule; meaning unchanged. |
| D-03 | 2026-11-10 | Three measurement boundaries (§2); `FULL_NATIVE_REFERENCE` = existing Native runtime unchanged; `RL_GPU_HEADLESS` = new batched runtime. | Specification §3, P1-0.4. |
| D-04 | 2026-11-10 | Physics batching lives in `robotic_cutting_sim.batched`; RL-facing code (vector env, reward, benchmark, telemetry, replay) in `robotic_cutting_sim.rl`; configs in `configs/rl_benchmark/`; scripts in `scripts/rl_ready/`; tests `tests/test_rl_*.py`. Existing modules are only extended where an opt-in hook is unavoidable. | Keeps frozen code untouched and reviewable. |
| D-05 | 2026-11-10 | WP-17 Native seed 512 / 15 recordings (actions, measured body poses, 121 physical samples) are the frozen replay regression inputs for material parity. | Same material input as Native without rerunning Kit; reproducible. |
| D-06 | 2026-11-10 | Batched physics architecture: ADR 0021 accepted with amendments — flat per-env concatenation packed from frozen single solvers, env-indexed copies of 13 kernels, grouped HashGrid, one CUDA graph per batch shape, masked partial reset, CPU bitwise parity (L0/L1) required, GPU invariants + outcome band (L2/L3). Workload A (batched ideal tracking, labelled `reduced_fidelity`) first; robot batching on Isaac Lab Newton N worlds is P1-1b under its own ADR. | xhigh analysis with source citations. |
| D-07 | 2026-11-10 | RCS GPU jobs: one per GPU under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock`, explicit device, ≥4 GB left free; rows labelled `SHARED_GPU`/`EXCLUSIVE_GPU` from sampled co-tenant use. | Shared machine. |
| D-08 | 2026-11-10 | Headless single-env and batched material runs use `.deps/isaacsim-venv/bin/python` without starting Kit (Python 3.12.12, Warp 1.17.0, NumPy 2.3.1, Torch 2.11+cu128 — identical to the WP-17 Native runs); CPU tests keep `.venv/bin/python -m pytest`; Torch-dependent tests run separately in the Torch venv and are reported as such. | Same numerics as Native; `.venv` has no Torch. |

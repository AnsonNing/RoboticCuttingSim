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

### Phase 2 — rubber material (in parallel with Phase 1 since 2026-11-10 14:05, decision D-12)

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
| D-09 | 2026-11-10 | Torch tests run as `PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python -m pytest -p no:cacheprovider …`; the overlay is a copy of the project `.venv` pytest packages (`scripts/rl_ready/make_pytest_overlay.sh`). Nothing is installed into `.deps`. | The Torch venvs have no pytest; `.deps` is read-only. |
| D-10 | 2026-11-10 | Robot batching design: ADR 0022 accepted — `IsaacLabBatchedCuttingEnv` on Newton/MuJoCo-Warp with N worlds at the origin (`env_spacing=0`, per-env `{ENV_REGEX_NS}` Frankas + tool, shared static workcell), batched LM IK, B1 rigid mount (`reduced_fidelity`) then B2 physical grasp with per-slot settled snapshot restore, per-substep graph-safe wrench kernel with the real frame chain, masked per-world reset. S0 CPU in parallel with workload A; GPU after M2. | xhigh analysis with CPU probes. |
| D-11 | 2026-11-10 | P1-6 acceptance contract `acceptance_v1` (R0): the external agentic optimizer accepts a candidate only through RCS's own versioned harness `scripts/rl_ready/run_perf_acceptance.py` (`perf_acceptance_v1`; one process per side and run, variant = `open_vector_env` kwargs, unknown/reserved options raise `UNSUPPORTED_OPTION`, `RCS_PERF_MARK` lines after a device synchronize, `result.json` `rcs_perf_result_v1` + npz dumps, `--out` never overwritten, at most one live device HashGrid per process) and judge `scripts/rl_ready/judge_perf_acceptance.py` (CPU, frozen evaluator checkout, recomputes physics from the dumps and the WP-17 Native records, `verdict.json` `rcs_perf_verdict_v1`, layers PHYSICS / PERFORMANCE / INTEGRITY), with thresholds, plan, seeds (dev 512, held-out 15), reference variant `{"particle_contact_mode": "alias_free_v1"}` and evaluator file set in `configs/rl_benchmark/acceptance_v1.json`. G9 passes on the caller's outer clock: N=16 paired median ≥ 1.05 with ≥ 4/5 pairs > 1.0, every N median ≥ 0.97, VRAM ≤ 1.10 × B + 256 MiB, inner/outer clocks within 5 %; reset p95 on the per-side medians over all ABBA processes at N: both sides `EXCLUSIVE_GPU` → FAIL if C > 1.25 × B + 2 ms; either side `SHARED_GPU` → PASS within 1.25 × B + 2 ms, WARN `shared_gpu_reset_bimodality` (not a failure) within 1.25 × B + 10 ms, FAIL beyond (slacks in `thresholds`; amended before any baseline after smoke_001 showed per-process reset modes of ~1 / ~7 ms independent of the variant, B 1.7 vs C 7.1 ms; evidence `p1_6_r0_003.json`). Freeze record (corrected 2026-11-10 after the Phase 1 acceptance review, finding 2; this record supersedes the in-file text "Frozen once committed" of `acceptance_v1.json`, which is historically inaccurate and is left unchanged because the file's bytes are now frozen): after its first commit `485241d`, `acceptance_v1` was amended in place four times, all before any baseline or candidate — `5ea4ce4` (position rule made descriptive), `3c2642a` (`contact_total_rel_tol` 0.01 → 0.02 after smoke_003), `214ac11` (reset p95 on a shared GPU FAIL → WARN after smoke_001 failed on reset), `ac3f6c6` (band pointer → band v2). Two of these (`3c2642a`, `214ac11`) loosened a gate after the data had been seen. `acceptance_v1` freezes at the Phase 1 merge into `main`, with `configs/rl_benchmark/acceptance_v1.json` sha256 `3592d9408f314c3d2825171361b8ec640534c60577bb6216a10c4f2b22897bf4`; merge commit: `40c94c756f5bd009b2532637dd40f92a51de9d1f` (PR #12, 2026-11-11). Any later change, including the finding-1 fix (G9 times only the pre-contact replay window at N=8/16), is `acceptance_v2`. `acceptance_v1` must not be used for a P1-6 acceptance (review §8). One material per process (each seed round is a child process). The P1-0 position band is a swappable, descriptive input (`physics.position_band`, WARN only) until the pre-declared band v2 replaces it. Amended 2026-11-10 before acceptance_v1 entered any baseline: `physics.position_band` now points at the pre-declared `configs/rl_benchmark/position_band_v2.json` (review finding F2; 10 frozen single-env replays per seed, one-sided 99 % log-normal prediction bound), still `descriptive`. `run_perf_acceptance_local.py` reproduces an acceptance without the optimizer. Report: `docs/reports/rl_ready/P1_6_R0_ACCEPTANCE_HARNESS.md`. | The optimizer must judge candidates with RCS's own frozen benchmark, never with the candidate's self-reported numbers; the previous runners hard-coded variants and wrote under the checkout. |
| D-12 | 2026-11-10 | User instruction (14:05,「可以平行」): Phase 2 (rubber) runs in parallel with Phase 1 instead of after the Phase 1 merge. Worktree `/home/ningan/RoboticCuttingSim-rl-rubber`, branch `claude/rl-rubber` from `claude/rl-ready-rubber-physics` @4c8bd03; order P2-0 → P2-6; built on the same batched material interface (no single-env-only rubber demo); each accepted step (xhigh review) is merged back into `claude/rl-ready-rubber-physics`. Phase 1 keeps priority on GPU slots; rubber work favours CPU tests and small GPU probes. The Phase 1 PR to `main` does not wait for Phase 2; Phase 2 reaches `main` in its own PR after its own acceptance. Material backend / constitutive model / near-incompressibility decisions go to rcs-xhigh first; implementation and tests to rcs-impl. | User instruction; round prompt "First steps" item 3. |
| D-13 | 2026-11-10 | Phase 1 xhigh acceptance review at `f03fbff`: **PASS_WITH_LIMITATIONS**, merge to `main` GO with conditions — the PR body carries the review's §7 limitation list and links the report; the PR head is `f03fbff` plus the report and documentation-only commits (code merged after `f03fbff`, including Phase 2 merges under D-12, needs its own review); findings 2, 6, 7 and 9 are corrected in the same PR as documentation. No-go for `acceptance_v1` in a P1-6 acceptance until findings 1 and 2 are resolved as `acceptance_v2`; findings 3 and 4 are fixed before B2 training-mode use; findings 5, 8, 10, 11 and 12 are follow-ups. After the merge the status is `MERGED_RCS_PENDING_OPTIMIZER_SYNC`. Report: `docs/reports/rl_ready/PHASE1_ACCEPTANCE_REVIEW.md`. | Review §1, §8. |
| D-14 | 2026-11-10 (window re-declared 2026-11-11) | P1-6 acceptance contract `acceptance_v2` (`configs/rl_benchmark/acceptance_v2.json`, schema `rcs_perf_acceptance_v1`, sha256 `b66ace2967b4aad8382ca4297032c0e513a4cf3ebe6564dad03494b8e7dd612a`; the first declaration `a008dd38…` timed steps 321–400 and was replaced before any candidate was judged, follow-up review finding 1 and coordinator decision (b)). **Why:** the Phase 1 xhigh acceptance review (findings 1 and 2) found that `acceptance_v1`'s G9 speed gate times only the pre-contact approach at N=8/16 (replay steps 21–120) and that `acceptance_v1` was amended in place four times while declared frozen. `acceptance_v1` is frozen at sha256 `3592d940…` (pinned), freeze commit `40c94c7` (the PR #12 merge into `main`, as recorded in D-11), and is never used for a real acceptance. **Window:** G9 replays s512 from reset and times control steps 229–400 (`warmup_steps` 228, `steps` 172, the same at every N; no `steps_by_num_envs`). Two counters: tool–material contact (stats slot 1; WP-17 Native s512 non-zero at 229–366, 101 steps) and particle–particle cut-face contact (slot 4; 315–364, 49 steps). The window times all 101 tool-contact steps, all 49 particle-contact steps and all labelling (0 → 37,958); only the approach (1–228) is untimed. `G9:window` gates, on the G6/G7 episode dumps of the same variants (not inside the G9 processes), ≥ 80 tool-contact steps (the harness now dumps slot 1 as `tool_contacts`) and ≥ 30,000 new labels per s512 env, set from Native with about 20 % margin before the re-check; particle contacts are reported, not gated. `performance.phases` reports per-phase step walls (candidate-reported, descriptive). **Statistics:** ABBA, outer clock, N=16 primary (median ≥ 1.05 and 3 of 3 pairs > 1), every-N floor 0.97; N {1, 16} and 3 pairs (v1: {1, 8, 16} and 5). This is a trade-off, not stricter than v1 overall: only the sign-test part is stricter (1/8 vs 3/16); with the 1.05 median floor both false-accept rates are about 0 for a neutral candidate; at pair noise σ ≤ 0.02 near-threshold errors roughly double, at σ ≥ 0.03 v2 loses power; the N=1 floor rests on 3 pairs of 172 steps (v1: 5 of 300); N=8 is not gated. Run acceptances on an `EXCLUSIVE_GPU` where possible (σ ≈ 0.006). **Budget (GPU time only):** a G9 run is 400 replay steps whatever the W/K split; from the EXCLUSIVE pre-contact rows G9 53.9 min (N=1 9.6 %), physics plan 44.5 min, 1.64 GPU-h exclusive per acceptance. Contact-phase factor (mean step over 1–400 / approach mean) measured at N=2 only, `alias_free_v1`, episode mode, two `SHARED_GPU` runs: 1.061 (foreign SM mean 31.5 %, contact-phase median 1.33× approach) and 1.033 (foreign SM 0 %, 1.01×), so about 1.69–1.75 GPU-h exclusive, within 2.5 h up to about ×1.43–1.48 co-tenant slowdown; largest lock hold (G9 N=16) about 50–52 min exclusive. The factor at N=16 is unmeasured. The G1/G2 CPU suites (about 50 min per side) add wall time, not GPU time. Unchanged from v1: physics plan, thresholds other than `primary_pairs_above_1`, seeds, reference variant, evaluator file set, `contact_total_rel_tol` 0.02 (smoke_003: frozen_v1 vs alias_free_v1 s15 totals differ by 0.51 %), band v2 descriptive, VRAM and reset p95 candidate-reported (reset: strict on EXCLUSIVE, WARN up to +10 ms on SHARED). GPU re-check (GPU0, N=2, s512, at `4943d75`): both envs 101 tool-contact steps in 229–400 (the same non-zero set as Native), 49 particle-contact steps, 0 → 37,958 labels. **Freezes at the commit that merges it into `main`** (`tests/test_acceptance_frozen.py`); any change after that is `acceptance_v3`. Reports: `docs/reports/rl_ready/P1_FOLLOWUP.md`, review `P1_FOLLOWUP_REVIEW.md`. Erratum (re-review of 994c679..0f50d76, finding 1): the descriptive text pinned inside `acceptance_v2.json` gives the s512 tool-contact gaps as 237–256 / 259–273 and s15's ranges as 229–516 (tool) / 321–842 (particle); the data say 237–257 / 259–274 and 230–517 / 322–843. No gate reads these numbers and the evidence carries the correct values; the config is not re-pinned because the GPU window re-check ran with `b66ace29…`. Freeze record: merged into `main` by PR #13 at merge commit `bfe6d5eaee0dd42ab6337acc8b6449f4b0da8eb0` (2026-11-11); `acceptance_v2.json` sha256 `b66ace2967b4aad8382ca4297032c0e513a4cf3ebe6564dad03494b8e7dd612a` is frozen there; any later change is `acceptance_v3`. | A speed gate that never times cutting cannot accept or reject a change to the cutting work; timing from the first tool contact costs no more than timing from the first particle contact (400 replay steps either way), and the budget forces fewer N and pairs. |

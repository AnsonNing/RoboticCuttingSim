# WP-12 Phase 6 — performance and profiling (spec §6.6)

Date: 2026-10-31 (KST). Branch `claude/wp12-physical-cut-surfaces` (HEAD `48b3326` plus an uncommitted tree).
Spec: [WP12_SPECIFICATION.md](../WP12_SPECIFICATION.md) §6.6. Readiness checks: [WP12_AGENTIC_READINESS.md](WP12_AGENTIC_READINESS.md).
Evidence: [`docs/evidence/wp12/phase6_readiness_001.json`](../evidence/wp12/phase6_readiness_001.json).

Nothing here changes physics, a cutting law, a goal evaluator or a threshold. Material parameters are provisional
and uncalibrated.

## 1. Profiles: what counts as physics acceptance

| Profile | Numerics | Status |
| --- | --- | --- |
| `full_physics_v1` (`configs/wp11/runtime_profiles_v1.json`) | 1 mm particles, 4 mm grid, 0.1 ms material dt, 20 ms control, scene-declared solver (`tlmpm_v5`, law 8) | **The only acceptance profile.** `acceptance_valid = true` in every reset info, evaluation and dataset row |
| `research_fast_v1` | 2 mm / 8 mm / 0.2 ms, wrist cameras only at 10 Hz, recording every 25 steps | **Development only**, `acceptance_valid = false`. The frozen 2 mm sever-location bound cannot pass at an 8 mm grid (its `fidelity_limits`; measured again in Phase 6: the location statistic is 3.0 mm) |
| Explicit coarse numerics (CPU tests: 4 mm particles, 0.5 ms) | no profile; numerics recorded in `physics_version` | **Development only** (tests, pipeline checks) |

No accelerated profile is used to claim a physics result in this report or in WP12_AGENTIC_READINESS.md.

## 2. Sources

| Source | Runtime | What it measures |
| --- | --- | --- |
| [WP12_SURFACE_LIVE.md](WP12_SURFACE_LIVE.md) §5 (`surface_live_006` A, `surface_live_002` B) | Native Isaac Sim 6.2, cuda:0 uncontended, CUT on the flat board, `full_physics_v1` numerics, seed 42, 740 control steps | A (`particle_surface_v1`) vs B (`cut_aware_surface_v1`) per-step split from the runner's `runtime_timing_trace.json` |
| [WP11_PHASE_H_PROFILING.md](WP11_PHASE_H_PROFILING.md) H2 | kinematic cuda:1 shared; Native TRIM (`native_reliability_trim_full_physics_001`) | `StageProfiler` stages; GPU kernel split (`--profile kernels`, CUDA activity timing) |
| Phase 6 runs (this report §4) | Native cuda:0 and kinematic cuda:1, both **shared with other agents' jobs at 100 % utilization** | `StageProfiler` stages (`--profile stages`) and the cut-aware per-extraction timings in `surface_provenance.json` |

## 3. Per-step split at the acceptance resolution (spec §6.6 items)

Native CUT, `full_physics_v1` numerics, cuda:0 **uncontended** (Phase 3 Layer 6 measurement; ms per 20 ms control
step, mean / p95 unless noted).

| §6.6 item | A `particle_surface_v1` | B `cut_aware_surface_v1` | Note |
| --- | --- | --- | --- |
| TLMPM physics (200 material substeps, CUDA graph + wait) | 269.7 / 274.7 | 270.4 / 274.1 | B's graph includes the recorder kernels |
| Cut-event update (cut_events_v1 recorder on the GPU) | — | ≈ +0.7 (inside run-to-run noise) | plus the host log fetch 0.8 / 0.8 |
| Surface extraction (excluding USD) | ≈ 45 (52.3 total − 7.0 upload − 0.9 world transform; re-meshes the moving particle cloud) | 225.0 (229.7 − 4.3 upload − 0.4): partition 12.6 (221.7 p95 when codes change), topology + QEF 18.5, quads 2.7, embedding 21.7, audit 38.1, reference-mesh validity + metadata 98.4, face-varying normals 30.2 | B is the NumPy/SciPy reference implementation on the host CPU |
| USD upload | 7.0 / 7.9 (`update`) | 4.3 / 4.5 (`update_face_varying_v1`) | B uploads ~4x fewer triangles (35.8 k vs 147 k) |
| Contact computation | inside the TLMPM step: blade SDF + particle hash-grid contacts are **80-82 %** of the GPU kernel time (WP-11 H2 kernel split, 261-268 ms GPU per step); Native grasp contact sampling 31.2 ms mean (WP-11 H2 Native TRIM) | same | the TLMPM row already contains the material contacts |
| Robot physics + IK + other | 84.1 (derived) | 81.9 (derived) | PhysX 10 substeps 17.6, IK 15.9 (WP-11 H2 Native TRIM) |
| Camera rendering (3 RTX cameras + 2 presentation views, every 10th step) | 384.3 per capture (≈ 38 per step amortized) | 367.7 per capture | Phase 2b/3 recording rate; the policy rate is `camera_every` |
| **Runner step total** | **406.1 / 612.5** | **582.0 / 795.4** | B ≈ +175 ms per step, all host CPU |
| Episode process wall | 393 s | 518 s | 740 steps + reset + recording |

Kinematic backend (no robot, no cameras), `full_physics_v1`, cuda:1 shared (WP-11 H2): the control step is
≥ 98.7 % TLMPM (`tlmpm_v5` TRIM 571 ms mean, SEVER 644 ms mean); oracle history 2.6-2.9 ms, recording 0.8 ms,
observation + validation + collision 0.95 ms per step. Episode level (TRIM v5): reset 1.1 s, evaluation 12.0 s,
recording finish 15.1 s, export 0.05 s.

## 4. Phase 6 measurements (GPU shared; contended)

Both GPUs were at 100 % utilization from other agents' jobs during every Phase 6 run (resolution study, E1 rows,
other users), so these wall times are **contended** and must not be read as the uncontended cost (§3 is). The
same measurement on a quiet GPU was not possible in this session: **NOT_RUN** (no quiet window). The per-step
times below come from the stored `actions.jsonl` rows (`material_wall_s`, `robot_physics_wall_s`, `step_wall_s`)
and the cut-aware per-extraction timings in `surface_provenance.json` (every control step + reset).

**Profiler caveat (kept, not hidden).** The runs were launched with `--profile stages`, but the runner's leak
watcher wrapped the env instance's `step`/`reset` with the methods bound before `StageProfiler.install()` patched
the class, so the profiler saw no control steps (`control_steps = 0`; only its episode-level stages are valid). The
watcher now resolves the class method at call time (fixed after these runs); the per-step numbers below do not
depend on the profiler.

### 4.1 Native, `full_physics_v1`, `cut_aware_surface_v1`, cuda:0 contended (`artifacts/wp12/phase6_004`)

ms per control step, mean / p95 (Native ep 0 CUT 700 steps, ep 1 TRIM 725 steps, ep 2 CUT 700 steps).

| §6.6 item | CUT ep 0 | TRIM ep 1 | CUT ep 2 | Uncontended reference (§3, B) |
| --- | --- | --- | --- | --- |
| TLMPM physics (material step incl. recorder kernels) | 985 / 1128 | 634 / 857 | 306 / 557 | 270 / 274 |
| Robot physics loop (PhysX substeps + coupling, includes the material step) | 1346 / 1669 | 1006 / 1308 | 617 / 846 | — |
| Cut-event update: host log fetch | 0.86 / 1.00 | 0.85 / 0.94 | 0.85 | 0.8 / 0.8 |
| Surface extraction B (excl. USD, incl. normals) | 247 / 482 | 285 / 504 | 234 / 451 | 225 |
| … partition / topology+QEF / embedding / audit / validity / normals | 16.2 / 20.6 / 23.1 / 41.3 / 107.0 / 31.9 | 30.1 / 19.6 / 22.9 / 48.4 / 105.9 / 51.4 | 15.9 / 19.3 / 22.0 / 38.9 / 100.7 / 30.5 | 12.6 / 18.5 / 21.7 / 38.1 / 98.4 / 30.2 |
| USD upload (`update_face_varying_v1`) | 4.68 / 4.89 | 4.59 / 4.85 | 4.45 | 4.3 / 4.5 |
| Control step total (policy env) | 1751 / 3650 | 1233 / 2724 | 703 / 883 | 582 / 795 |
| Extractions (audit errors) | 701 (0) | 726 (0) | 701 (0) | — |

The host-CPU surface stages are close to the uncontended values (they do not use the GPU); the GPU-bound material
step is 2.3–3.6x slower than uncontended. Camera rendering (policy cameras every 5th step, `camera_every = 5`) and
the grasp contact sampling are not separable in these rows; their uncontended values are in §3.

Episode level (ep 0 / 1 / 2, from the profiler's episode stages, which are valid): Native reset 10.9 / 6.8 / 6.7 s
(ep 1 and 2 follow a task switch, `reload_in_place`), evaluation 2.3 / 5.8 / 1.9 s, recording finish 14.4 / 15.2 /
13.5 s, dataset export 5.6 / 6.8 / 5.3 s. Episode walls 1271 / 940 / 530 s: ep 2 ran while the other GPU jobs had
eased (its material step 306 ms is close to the uncontended 270 ms), which shows how much of ep 0/1 is contention.

### 4.2 Kinematic, `full_physics_v1`, cuda:1 contended (`artifacts/wp12/phase6_005`)

CUT (700 steps): control step 485 / 556 ms, material step 481 / 553 ms (99.2 %), cut-aware surface (recorded frames
only, 71 extractions) 272 / 488 ms per extraction; reset 0.21 s, evaluation 7.1 s, recording finish 13.0 s, export
0.04 s; episode wall 381 s. TRIM (725 steps): control step 401 / 553 ms, material step 397 / 549 ms, surface 292 / 496 ms per
extraction (74), evaluation 5.5 s, recording finish 14.1 s; episode wall 334 s. Both PASS (CUT 0.50 / 0.50; TRIM
dome removed 0.979), `acceptance_valid = true`, task ids equal to the Phase 2b records.

### 4.3 Kinematic, `research_fast_v1` (development only, `artifacts/wp12/phase6_003`)

CUT 700 steps: control step 152 / 161 ms, material step 149 / 158 ms. That is 3.2x faster than `full_physics_v1`
on the same contended GPU, at numerics that cannot pass the frozen sever-location bound.


## 5. Reading

- `full_physics_v1` remains the acceptance profile. No Phase 6 claim uses `research_fast_v1` numbers for physics.
- At the acceptance resolution the TLMPM step dominates the kinematic control step (≥ 98.7 %); inside it the
  blade/particle contact kernels are ~80 % of the GPU time (WP-11 H2). The cut-event recorder adds no measurable GPU
  time (Phase 3 Layer 6).
- On Native the cut-aware surface is the largest host cost after the material step: ~225 ms per step uncontended
  (≈ 240–280 ms here), of which the reference-mesh validity check (~100 ms) and the audit (~40 ms) are diagnostics
  that a production profile could run only at recorded frames. That would be a new, versioned surface option; it is
  not done here. The default `particle_surface_v1` path is unchanged.
- Camera rendering is 368–384 ms per RTX capture of 3 cameras (+ presentation views); its per-step cost is set by
  `camera_every` / `record_every`, both recorded with every episode.
- Contention: a shared GPU made the Native control step 2–3x slower than its uncontended value without changing
  any verdict (§4.1 vs WP12_SURFACE_LIVE §5 and WP12_AGENTIC_READINESS §5.2).


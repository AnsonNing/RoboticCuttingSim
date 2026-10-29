# WP-11 Phase H: Native reset memory, profiling, safe optimizations, runtime profiles

**Verdicts at a glance**

| Item | Verdict |
| --- | --- |
| H1 diagnosis: where the per-reset Native host RSS growth comes from | **DONE**: Kit `open_stage` of the workcell retains 55-66 MiB per open (no Python objects, no USD layers); ~0.04 MiB/reset Python heap |
| H1 fix (opt-in `stage_reset="reload_in_place"`) | **PASS** for memory and reset neutrality; limited to `camera=True` (camera-off in-place reset crashes PhysX, 4 of 4 probes) |
| H1 Test F rerun, clean worktree `ecc77de`, `full_physics_v1`, v1a gates | **NOT POLICY_READY**: only `gpu_memory_no_leak` FAILS (52 MiB/episode on 3 non-monotonic points, gate 32). Host memory now PASSES (56.5 MiB/episode, gate 64; Phase G 140) |
| Same run scored with v1 | NOT POLICY_READY: `tool_at_rest_after_reset` (all 4, the known PhysX clamp jitter) and `gpu_memory_no_leak` |
| H2 profiling (kinematic TRIM/SEVER x `tlmpm_v4` law 8 / `tlmpm_v5`, Native TRIM) | **DONE** (tables below) |
| H3 safe optimizations | 1 implemented, opt-in (`surface_update="on_capture"`), physics-neutral by construction; kinematic has no non-solver overhead above 1% to remove |
| H4 runtime profiles `full_physics_v1` / `research_fast_v1` | **DONE**, opt-in, tested; kinematic speedups 8.0x (TRIM) and 9.6x (SEVER) |
| CPU tests | targeted 23 + 14 + 9 PASS; full suite **1035 passed, 5 xfailed, 0 failed** (rc 0, 1436 s) |

Material parameters are provisional and uncalibrated. Nothing here changes a cutting law,
threshold, frozen solver (`tlmpm.py`, `tlmpm_v2.py`, v4/v5 numerics), `final_v4` behaviour or a
v1/v2 task threshold. Every new behaviour is opt-in; without `runtime_profile`, `stage_reset`,
`surface_update` or `--profile` every path is the previous one.

| Item | Value |
| --- | --- |
| Evidence | [`docs/evidence/wp11/phaseH_profiling_001.json`](../evidence/wp11/phaseH_profiling_001.json), written read-only by `scripts/summarize_phase_h.py` |
| Commits | `a5b05b8` (stage reset option, profiler, profiles, collector flags, probe), `43223f9` (episode plans), `ecc77de` (in-place physics via stage-update interface; cameras required), `be047b7` (`surface_update`), `4b5b2e1` (CUDA activity timing in kernels mode) |
| Clean worktrees (`git_dirty = false`) | `/mnt/data/ningan/RoboticCuttingSim-wp11-H-43223f9` (kinematic stage profiles), `-ecc77de` (Native smoke + Test F), `-4b5b2e1` (kernel split, Native speed runs). Each links `.venv`, `.deps/*`, `artifacts/workcell_asset_cache` and `artifacts/wp11` to the main checkout. |
| Devices | Native: GPU 0 (RTX 5090), launched only with <= 2 other active jobs (project rule `gpu_other_jobs`) and >= 12 GB free; one Isaac job of mine at a time. Kinematic: `cuda:1`, shared with other agents' jobs at up to 100 % utilization; wall times include that contention. |
| Runtimes | Isaac Sim 6.2 Native (Kit/PhysX CPU articulation, RTX wrist D405 x2 + global D455), Warp 1.17.0, `.venv` (CPU tests, kinematic) |
| Diagnostic probes | `artifacts/wp11/phaseH/h1_probe_*` (diagnostic only; the H1 probes ran from the main checkout and say `git_dirty = true`) |

## H1. Native reset memory

### Diagnosis

`scripts/phaseH_native_memory_probe.py` opens the `cutting_research_v1` env on Native with the
Phase G task (`trim01_wp11_workcell` / `trim_bump_rho10`, planner options as Phase G, cameras on,
`camera_every` 5) and runs N resets of 5 HOLD steps in one process. Per reset it records RSS at
reset sub-stages (read-only wrappers), after the steps, after `gc.collect()` and after
`malloc_trim(0)`, the number of loaded `Sdf.Layer`s, gc object counts by type and optionally
`tracemalloc` diffs. The stage modes isolate Kit: after one normal reset they only export and/or
(re)open the workcell. Slopes are least-squares over resets >= 1 of the RSS after `malloc_trim`.

| Probe (`artifacts/wp11/phaseH/…`) | What runs per iteration | Host RSS slope (MiB/reset) |
| --- | --- | --- |
| `h1_probe_baseline_002` | full env reset, legacy `reopen`, cameras on (6 resets) | **81.5** (Phase G collector diagnostic: 82.0) |
| `h1_probe_camera_off_001` | same, cameras off | 78.4 (cameras are not the cause) |
| `h1_probe_stage_same_path_001` | `open_stage` of the same workcell file only | **61.1** |
| `h1_probe_stage_new_path_001` | export a new workcell copy + open it (the legacy reset's stage path) | 65.9 |
| `h1_probe_stage_close_open_001` | `close_stage` + `open_stage` | 54.9 |
| `h1_probe_export_only_001` | workcell export only | 0.1 |
| `h1_probe_stage_new_empty_001` | `new_stage()` (empty) | 2.9 |
| `h1_probe_stage_reload_001` | clear session layer + reload root layer, no open | -27.4 (no growth) |
| `h1_probe_reload_in_place_001` | full reset, in-place v1 (`IPhysx.release_physics_objects`) | 17.5 |
| `h1_probe_reload_in_place_tracemalloc_001` | in-place v1 + undo-stack clearing, `tracemalloc` (8 resets) | Python traced heap **+0.04 MiB/reset** |
| `h1_probe_reload_v2_camera_on_001` | in-place v2 (stage-update `reset_simulation`/detach/attach) | 11.2 |
| `h1_probe_reload_v3_camera_on_001` | in-place v3 = committed (v2 + `gc.collect`) (8 resets) | **14.0** |

Other findings in the same probes:

- The loaded `Sdf.Layer` count is constant across resets in every mode (92 with cameras, 91 without):
  no old stage is retained.
- gc-tracked Python objects grow by 26 per reset in the legacy reset (7 `pxr.Sdf.Layer` wrappers,
  3 SyntheticData `NodeConnectionTemplate`, weakrefs); the Python heap measured by `tracemalloc`
  grows by 0.04 MiB/reset. The leak is native, not Python.
- `malloc_trim(0)` returns 100-175 MiB after every reset (freed heap held by glibc) but does not
  stop the growth.
- The first in-place version left the Kit undo history growing (+20 `omni.kit.undo` entries per
  reset); `open_stage` clears it, so the in-place reset now clears it too.

**Attribution.** Of the ~82 MiB per legacy reset, 55-66 MiB is retained inside Kit by `open_stage`
of this workcell (same file, new file or close+open; ~3 MiB for an empty stage; 0 for the export),
i.e. it scales with the stage content and lives in Kit/USD/Hydra/PhysX native memory that our code
does not own. The remaining ~15-25 MiB per reset is also native (physics, sensors, material, render
products are re-created each reset); it is not attributed further.

### Fix: `IsaacSimCuttingConfig.stage_reset = "reload_in_place"` (opt-in)

After the first reset (which opens the stage as before), a reset with the same workcell export
arguments keeps the open stage and:

1. detaches every replicator annotator and destroys the render products of the previous reset;
2. drops the Python tensor-view / prim wrappers, `gc.collect()`, invalidates the SimulationManager
   views, and on the SimulationManager stage-update interface runs `reset_simulation`, `on_detach`,
   then (after step 3) `on_attach`, as a stage close/open does;
3. `renew_stage_layers`: clears the session layer (every reset edit is authored there) and reloads
   the root layer and any other dirty file layer from disk (unit-tested to give exactly the
   composition of a fresh open);
4. replays the SimulationManager stage-opened reset and clears the Kit undo stack;
5. continues with the unchanged reset code (fixture, tool, grasp settle, cameras, material).

The default stays `"reopen"`. `full_physics_v1` and `research_fast_v1` select `reload_in_place`.

**Restriction.** With `camera=False` the second reset segfaults inside PhysX `start_simulation`
for every release strategy tried (v1, v2, v3 and USD notices only: 4 of 4 probes,
`h1_probe_reload_*camera_off*`, logs preserved). With cameras on, 0 crashes in 4 probes
(29 in-place resets) plus 7 collector resets and 3 Test F resets. `reload_in_place` therefore requires
`camera=True` and `physical_grasp` (config `ValueError` otherwise). The cause was not found.

### Neutrality

- **CPU**: the kinematic path is untouched by the stage reset. The profiler is bitwise neutral
  (`test_profiler_is_bitwise_neutral_on_cpu`: positions, velocities, codes, per-step statuses and
  commanded edges, and the evaluation are identical with no profiler, stage mode and kernels mode).
- **Native, collector diagnostic** `native_memory_probe_inplace_001` (8 x 20 HOLD steps, the
  Phase G diagnostic plan, `ecc77de`, `git_dirty = false`): every post-reset check of episodes 1-7
  (tool and edge pose, active joints, material codes hash, tool linear/angular speed, all 20 Phase G
  reset checks) is **bitwise identical** to episode 0, and the setup receipts differ only in the
  PhysX step counter. An in-place reset (episode 5) is bitwise identical to a Phase G `reopen`
  reset (`native_memory_probe_001` episode 3) in all 20 reset checks, and after 20 control steps
  the measured `q`, `qdot`, tool, hand, tool body and finger states and final codes are bitwise
  identical as well.
- **Native, Test F**: every reset of episodes 1-3 equals episode 0 bitwise (same checks), the grasp
  re-stabilized (`PASS_STABLE_GRASP`) every reset, and the scripted repeat (episode 3) reproduced
  episode 0's final cut codes exactly (sha256 `cd4631b3…`; Phase G's repeat differed).

### Memory after the fix

| Run | Episodes | Host RSS after episode (MiB) | Host slope | GPU after episode (MiB) | GPU slope |
| --- | --- | --- | --- | --- | --- |
| Phase G `native_memory_probe_001` (reopen) | 8 x 20 steps | 5951 … 6593 | 82.0 | 6021 … 6137 | -45.6 |
| Phase H `native_memory_probe_inplace_001` | 8 x 20 steps | 6086, 6207, 6251, 6276, 6293, 6297, 6353, 6358 | **24.2** | 6045, 6173, 6174, 6174, 6174, 6238, 6174, 6238 | 9.3 |
| Phase G `native_reliability_trim_001` (reopen) | 4 full | 10974, 11292, 11407, 11573 | 140.3 | 6205, 6336, 6274, 6308 | -14.0 |
| Phase H `native_reliability_trim_full_physics_001` | 4 full | 15243, 15352, 15406, 15465 | **56.5** | 5690, 5782, 5607, 5886 | **52.0** |

The fix is not "no leak": Kit still retains memory per `open_stage` (we no longer call it after the
first reset), the in-place reset still grows ~14-24 MiB per short episode, and a full episode adds
~55 MiB. Long collections should still expect slow host growth; process recycling every N episodes
was **not implemented** (NOT_RUN) and remains the recommended mitigation for runs of hundreds of
episodes.

### Test F rerun (Native, clean worktree)

Command (from `/mnt/data/ningan/RoboticCuttingSim-wp11-H-ecc77de`, `git_dirty = false`, GPU 0,
started 2026-10-30 07:41:55 KST, 2783 s wall, exit 3 = not POLICY_READY):

    OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python \
      scripts/collect_dataset.py --runtime isaac_sim --gpu 0 --scene trim01_wp11_workcell --goal trim_bump_rho10 \
      --planner-options '{"max_rotation_deg": 1.0, "max_translation_m": 0.001}' \
      --episode-plan configs/wp11/phaseG/native_reliability_trim_v1.json --runtime-profile full_physics_v1 \
      --camera-every 5 --record-every 10 --profile stages \
      --reliability-config configs/wp11/phaseG_reliability_v1a.json --reliability-config configs/wp11/phaseG_reliability_v1.json \
      --out artifacts/wp11/phaseH/native_reliability_trim_full_physics_001

Same plan, task (`CFG-trim01_wp11_workcell-trim_bump_rho10-570db2f5`), seeds (42) and gates as
Phase G; the differences are `full_physics_v1` (formal resolution, `reload_in_place`) and the opt-in
stage profiler (declared; it only wraps methods with timers).

| Episode | Controller | Expect | Verdict | Dome removed | Home | Reset wall | Episode wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | scripted, measured-edge feedback | PASS | PASS | 0.9770 | PASS | 75.9 s (first open) | 824 s |
| 1 | hand-written action program | – | PASS | 0.9856 | PASS | 7.2 s | 639 s |
| 2 | deliberate failure (stop after PRE_CUT) | FAIL | FAIL (dome 0) | 0.0 | PASS | 7.4 s | 554 s |
| 3 | scripted repeat | PASS | PASS | 0.9770 (codes = episode 0) | PASS | 7.9 s | 715 s |

`phaseG_reliability_v1a` (primary): every per-episode check passes, consecutive episodes PASS, host
memory PASS (56.5 <= 64), repeatability PASS, failed episode stored and usable PASS, **GPU memory
FAIL (52.0 > 32 MiB/episode)** -> **POLICY_READY = false**. The GPU fit uses 3 points (5782, 5607,
5886) that go down and then up, and the 8-episode in-place diagnostic gives 9.3 MiB/episode, so this
looks like allocator noise rather than a leak, but the predeclared gate fails and the verdict stands.
Under v1 the run also fails `tool_at_rest_after_reset` on all four episodes (PhysX clamp jitter, see
the v1a amendment). A longer declared run (>= 8 full episodes) is needed to settle the GPU trend.

Disk floor: the grasp-trace JSONL files under `runtime/` of the H1 probes, the Native diagnostic and
this run were gzip-compressed in place after they were complete (lossless; sha256 of every original
in `artifacts/wp11/phaseH/*_grasp_trace_sha256_before_gzip.txt`) to keep >= 15 GB free on /mnt/data.

## H2. Profiling

`robotic_cutting_sim.profiling.StageProfiler` (opt-in, `scripts/collect_dataset.py --profile`).
Inclusive stage times per control step (`PolicyReadyEnv.step`); "share" is of the summed
control-step time. `tlmpm_substeps_graph` is the CUDA-graph launch of the 200 material substeps plus
the device wait. Overhead: research-fast TRIM with and without the profiler 53.3 s vs 53.2 s wall.

### Kinematic, formal resolution (1 mm, 4 mm grid, 0.1 ms), `cuda:1` shared, seed 42, full episode

| Run | Solver | Steps | Control step mean / max (ms) | TLMPM substeps (ms, share) | Oracle history | Recording | Obs + validation + collision | Episode wall | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kin_trim_v5_full_stages_001` | `tlmpm_v5` law 8 | 700 | 571 / 836 | 564.2, 98.8 % | 2.9 ms, 0.50 % | 0.84 ms, 0.15 % | 0.95 ms, 0.17 % | 428 s | PASS, dome 0.9789 |
| `kin_trim_v4_full_stages_001` | `tlmpm_v4` law 8 | 700 | 635 / 840 | 628.7, 99.0 % | 2.8 ms, 0.43 % | 0.79 ms, 0.12 % | 0.95 ms | 470 s | PASS, dome 0.9789 |
| `kin_sever_v4_full_stages_001` | `tlmpm_v4` law 8, K=2 | 400 | 489 / 563 | 482.9, 98.7 % | 2.7 ms, 0.54 % | 0.78 ms | 0.94 ms | 209 s | PASS, 0.50 / 0.50 |
| `kin_sever_v5_full_stages_001` | `tlmpm_v5` law 8, K=2 | 400 | 644 / 825 | 637.1, 99.0 % | 2.6 ms | 0.77 ms | 0.96 ms | 278 s | PASS, 0.50 / 0.50 |

Episode-level (TRIM v5): reset 1.1 s, evaluation (topology) 12.0 s, recording finish (compressed
physics history) 15.1 s, export 0.05 s, policy (planner) 0.12 ms per call. Slot check 0.18 ms and
label/stroke readback 0.15 ms per step.

### Kinematic kernel split (first 150 steps, kernels mode, CUDA activity timing)

`--profile kernels` launches the identical kernels eagerly (graph disabled) and records each
kernel's GPU execution time with `warp.timing_begin/timing_end`. GPU kernel time per control step:

| Run | GPU ms/step | Contacts | P2G | G2P (+MUSL, F update) | Slot claim | Grid update | Label/stroke | Stress | Memset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kin_trim_v5_kernels_002` | 267 | **80.2 %** (`_rigid_contacts` 54.8, `_particle_contacts` 24.5) | 6.7 % | 4.9 % | 1.9 % | 1.2 % | 2.3 % | 0.9 % | 1.9 % |
| `kin_trim_v4_kernels_002` | 261 | 82.2 % | 7.3 % | 4.5 % | – | 0.7 % | 2.3 % | 0.9 % | 1.9 % |
| `kin_sever_v4_kernels_002` | 266 | 81.6 % | 7.3 % | 4.7 % | – | 1.3 % | 2.4 % | 0.9 % | 1.9 % |
| `kin_sever_v5_kernels_002` | 268 | 80.5 % | 6.4 % | 4.7 % | 1.9 % | 1.2 % | 2.4 % | 0.8 % | 1.9 % |

The blade SDF contact over every particle and the particle-particle hash-grid contact dominate;
P2G/G2P/grid/slot claim together are ~13 %. The hash-grid build itself is not a Warp launch and is
not in this table. `kin_trim_v5_kernels_001` (first attempt: device sync before and after every
launch on the shared GPU) was dominated by launch+sync latency (0.7 ms per one-thread kernel) and
is kept but superseded; `kin_trim_v4_kernels_001` was stopped by me during that attempt (`ABORTED.txt`).

### Native live TRIM (`native_reliability_trim_full_physics_001`, episode 0, `full_physics_v1`)

700 task steps (plus 335 return-home steps reported separately), mean control step **669 ms**
(max 1908 ms); episode wall 824 s.

| Stage | Mean per control step (ms) | Max (ms) | Share |
| --- | --- | --- | --- |
| TLMPM substeps (graph + wait) | 389.0 | 585 | 58.1 % |
| Camera render + readback (RTX, 3 cameras; 140 of 700 steps, 423 ms per render) | 84.6 | 634 | 12.6 % |
| Surface extraction + USD mesh update (every step) | 78.7 | 339 | 11.8 % |
| Grasp contact sampling (10 substeps, contact sensor + JSONL trace) | 31.2 | 782 | 4.7 % |
| PhysX step (10 substeps) | 17.6 | 24 | 2.6 % |
| Proprioception readback (PhysX tensors) | 16.3 | 529 | 2.4 % |
| IK (Newton LM) | 15.9 | 32 | 2.4 % |
| Recording IO (EpisodeRecorder step + frames) | 13.3 | 146 | 2.0 % |
| Observation (deep copies incl. camera arrays) | 3.4 | 6 | 0.5 % |
| Oracle history | 1.9 | 16 | 0.3 % |
| Camera extrinsics | 1.6 | 11 | 0.2 % |
| Material finite/det(F) check readback | 1.0 | 3 | 0.2 % |
| Action validation + collision check | 0.7 | 5 | 0.1 % |

Episode level: reset 76.2 s on the first reset (69.8 s of it inside the runner's stage load and
authoring) and 7.2-7.9 s for the in-place resets; return home 227 s for 335 steps; recording finish
26.3 s; dataset export 14.1 s; evaluation 11.0 s. In-place episodes 1 and 3: mean control step 608
and 630 ms with the same split.

## H3. Safe optimizations

**`surface_update="on_capture"` evidence** (`native_speed_trim150_full_physics_001` vs
`…_on_capture_001`, same worktree `4b5b2e1`, seed 42): the measured robot joints are bitwise
identical on every one of the 150 steps (also identical to Test F episodes 0 and 3), and the final
codes are identical. Material positions differ by up to 1.3e-5 m between the two runs; that is the
run-to-run GPU (atomic) nondeterminism of the material, present with either mode. Recorded depth
frames differ by <= 3.6e-5 m (one silhouette pixel 0.17 m) and RGB by up to 70/255; the RGB of
`wrist_d405_1`, which does not see the specimen, also differs between runs (RTX is not run-to-run
deterministic). The physics path is bitwise unchanged; frame equality cannot be shown bitwise on
Native because RTX and the GPU material are not deterministic across processes.

| Candidate | Measured cost | Decision |
| --- | --- | --- |
| Native surface extraction every control step while frames render every 5 steps | 78.7 ms/step, 11.8 % | **Implemented, opt-in**: `IsaacSimCuttingConfig.surface_update="on_capture"` (`--surface-update on_capture`) re-extracts the visual `ContinuousSpecimen` mesh only before a camera render (and for evaluation, as before). The mesh has no physics API (test asserts the bridge authors none), so physics is unchanged by construction; every rendered frame still shows the surface extracted at its own time step. Default unchanged; the v1 profiles were not changed after being used. Measured below. |
| Kinematic overhead outside the TLMPM substeps | <= 1.2 % of the step at 1 mm | nothing worth removing; the oracle history readback (0.5 %) is needed by the goal evaluator |
| Narrow the Native per-step `debug_state()` readback | 1.0 ms/step, 0.2 % | not done (below the noise) |
| RTX render warm-up per capture (>= 5 orchestrator steps) | 423 ms per render | not done: fewer subframes change the image content (not bitwise); candidate for a future camera profile |
| Contact kernels (80 % of solver GPU time) | – | out of scope: solver numerics of frozen versions are not changed; a culled contact pass would be a new versioned solver |

## H4. Runtime profiles

`configs/wp11/runtime_profiles_v1.json` (declared before any profile run) and
`runtime_profiles.py`; selector `open_cutting_env(..., preset="cutting_research_v1",
runtime_profile=...)` and `collect_dataset.py --runtime-profile`. See
[POLICY_READY_API.md](../POLICY_READY_API.md#runtime-profiles-wp-11-phase-h4-opt-in) for fields.

- `full_physics_v1` (`acceptance_valid: true`): requires 1 mm / 4 mm / 0.1 ms / 20 ms, refuses caller
  numerics (`particle_spacing_m`, `material_dt`, `dx`, `history_stride`), Native `reload_in_place`,
  cameras as declared, recording every 10 steps.
- `research_fast_v1` (`acceptance_valid: false` on every reset info, evaluation, episode and index
  row): F3 'coarse' numerics (2 mm particles, 8 mm grid via a recorded scene-solver override, 0.2 ms
  material dt), wrist cameras only at 10 Hz, recording every 25 steps. Fidelity limits (F3): TRIM
  removal 0.816 vs 0.956, CUT location 3.0 mm (ideal floor) vs 1.5 mm, contact forces not converged.

### Speedups (same task, seed 42, scripted controller)

| Runtime | Task | `full_physics_v1` | `research_fast_v1` | Speedup | research_fast outcome |
| --- | --- | --- | --- | --- | --- |
| Kinematic, `cuda:1` | TRIM `trim_bump_rho10` (v5), full 700 steps | 428 s wall, 571 ms/step (97 044 particles) | 53.3 s wall, 70.9 ms/step (12 136 particles) | **8.0x** wall, 8.1x per step | PASS, dome 0.912 (formal 0.979) |
| Kinematic, `cuda:1` | SEVER `sever_mid` (v4, K=2), full 400 steps | 209 s, 489 ms/step | 21.8 s, 47.7 ms/step | **9.6x** wall, 10.3x per step | **FAIL**: two 50/50 pieces but `cut_at_declared_planes` fails (the F3 location limit) |
| Native, GPU 0 | TRIM, first 150 steps (`native_speed_trim150_*`) | 125.5 s episode wall, 708.8 ms/step (TLMPM 447, render 98.8, surface 49.0) | 55.5 s, 276.2 ms/step (TLMPM 60.3, render 95.4, surface 22.2) | **2.26x** wall, **2.57x** per step | not evaluated (150-step budget) |
| Native, GPU 0 | TRIM, first 150 steps, `full_physics_v1` + `surface_update=on_capture` (H3) | 125.5 s, 708.8 ms/step | 113.9 s, 633.2 ms/step (surface 8.8 ms/step instead of 49.0) | **1.10x** wall, 1.12x per step | robot `q` bitwise identical on all 150 steps; final codes identical |

## Commands

Kinematic stage profiles (clean worktree `…-H-43223f9`, `DEV=cuda:1`, sequential queue; logs
`artifacts/wp11/logs/phaseH_kin_*.log`, queue log `phaseH_kin_queue_001.log`):

    TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/collect_dataset.py --runtime kinematic \
      --scene trim01_wp11_workcell --goal trim_bump_rho10 --planner-options '{"max_rotation_deg": 1.0, "max_translation_m": 0.001}' \
      --episode-plan configs/wp11/phaseH/profile_kin_scripted_v1.json --runtime-profile full_physics_v1 --profile stages \
      --device cuda:1 --out artifacts/wp11/phaseH/kin_trim_v5_full_stages_001
    # same with --runtime-profile research_fast_v1 (kin_trim_v5_fast_stages_001; without --profile: _fast_noprofile_001),
    # --scene configs/wp11/phaseH/scenes/trim01_wp11_workcell_phaseH_v4.json (v4 copy), --scene bar_160x30x20 --goal sever_mid,
    # --scene configs/wp11/phaseH/scenes/bar_160x30x20_phaseH_v5.json --goal sever_mid

Kernel split (worktree `…-H-4b5b2e1`, queue log `phaseH_kin_queue_002.log`): the same with
`--episode-plan configs/wp11/phaseH/profile_kin_kernels_v1.json --profile kernels` -> `kin_*_kernels_002`.

Native memory diagnostic (worktree `…-H-ecc77de`, 8 x 20 steps, 4.4 min):

    OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python scripts/collect_dataset.py \
      --runtime isaac_sim --gpu 0 --scene trim01_wp11_workcell --goal trim_bump_rho10 --planner-options '…' \
      --episode-plan configs/wp11/phaseG/native_memory_probe_v1.json --camera-every 5 --record-every 10 \
      --stage-reset reload_in_place --reliability-config configs/wp11/phaseG_reliability_v1a.json \
      --out artifacts/wp11/phaseH/native_memory_probe_inplace_001

H1 probes (main checkout, diagnostic):

    OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python scripts/phaseH_native_memory_probe.py \
      --gpu 0 --resets 6|8 --steps 5 [--camera off] [--mode stage_same_path|stage_new_path|stage_close_open|export_only|stage_new_empty|stage_reload] \
      [--stage-reset reload_in_place] [--tracemalloc 4] --out artifacts/wp11/phaseH/h1_probe_<name>

Native speed runs (worktree `…-H-4b5b2e1`, `/mnt/data/ningan/tmp/…/scratchpad/native_speed.sh`, queue log
`phaseH_native_speed_queue_002.log`): `--episode-plan configs/wp11/phaseH/native_speed_trim_150_v1.json
--profile stages` with `--runtime-profile full_physics_v1 --camera-every 5 --record-every 10`, with
`--runtime-profile research_fast_v1`, and with `full_physics_v1 … --surface-update on_capture`.

Tests: `TMPDIR=/mnt/data/ningan/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_phaseH_runtime.py`
(14 passed), `tests/test_policy_ready_env.py` (14 passed), `tests/test_isaac_sim_runner.py` (9 passed);
full suite `.venv/bin/python -m pytest -q -p no:cacheprovider` -> 1035 passed, 5 xfailed, rc 0, 1436 s
(log `artifacts/wp11/logs/phaseH_cpu_full_001.log`; started on the tree at `be047b7`; `profiling.py`
changed to `4b5b2e1` while it ran, and `tests/test_phaseH_runtime.py` was rerun at `4b5b2e1`: 14 passed).

## Open items

- Native POLICY_READY: GPU memory gate fails on 3 noisy points; run a declared >= 8-episode full
  Native reliability run to settle the GPU and host trends.
- Residual host growth with `reload_in_place` (~15-25 MiB per short episode, ~55 MiB per full
  episode) is unattributed native memory; process recycling every N episodes is NOT_RUN.
- `reload_in_place` with `camera=False` segfaults in PhysX `start_simulation`; cause unknown, the mode
  is refused without cameras.
- The first Native reset of the Test F run took 76 s (fresh worktree, cold Kit caches); the later
  speed runs from another fresh worktree took 11-12 s and Phase G 15 s. Not investigated further.
- research_fast_v1 on Native is only 2.3x faster: rendering (about 0.48 s per capture for two
  cameras vs 0.49 s for three) and the PhysX/IK/contact-sensor loop do not scale with the material
  resolution. A cheaper camera profile would need a declared rendering change.
- Material parameters are provisional and uncalibrated.

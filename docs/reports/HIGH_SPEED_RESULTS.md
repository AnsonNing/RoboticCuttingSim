# High-speed simulation research

The [benchmark receipt](../evidence/high_speed/benchmark.json) contains the measured rows,
raw artifact paths and source identities. Original Rubber receipts are also
available at milestone `d13e05a`.

## Initial Rubber timing

The fresh Reference run on physical GPU 1 used RTX 5090, driver 570.211.01,
Warp 1.17.0, Torch 2.11.0+cu128 and one OpenBLAS thread. It held the GPU1
lock. Compute-PID samples showed no other compute jobs.

| Profile | N | Grid | Substeps per 20 ms | Warm batch wall | Aggregate env-steps/s | Sampled process MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rubber_rl_env_v2`, full slots, CUDA Graph | 8 | 1 mm | 5226 | 8.371 s | 0.956 | 1228 |

The five timed steps ran during the seeded preload and were finite with zero
invalid bits. This row gives early throughput feedback. Its scratch wrapper
then failed when the reduced profile rejected full-slot geometry. The quick
result and the failure remain recorded. The longer runs time 50 steps after
the 10-step preload. They record full-slot Reference separately and compare
Reference and the existing reduced profile with identical alternating
`L100_t4` and `L150_t4` slots and seeds.

The benchmark receipt records the quick row and the scratch wrapper error.

```bash
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/perf/benchmark_rubber.py \
    --slots paired --num-envs 8 --gpu-index 1 --out artifacts/rubber_fresh_pair
```

Add `--resolution rubber_rl_v1` for the existing reduced profile, or use
`--slots full` for the full-slot Reference. The runner records synchronized
step walls, cold build/reset timing, table contact, invalid bits and GPU
compute co-tenants. Each invocation requires a fresh output directory.

## Timestep bucketing estimate

The Reference wave-speed rule applied to 8,192 reset seeds gives 4,030
median per-environment substeps versus the fixed 5,226. Random groups of
eight average 4,755.88 substeps at the within-group maximum, a 9.0% reduction.
Perfectly sorted groups of eight average 4,005.29, a 23.4% reduction. These
are analytical estimates before deformation, independently timed resets,
and graph scheduling. Bucketing stays outside the first implementation.


## Full-slot Reference drag

50 synchronized steps after the 10-step seeded preload. All slots recorded table
contact in all 50 steps and zero invalid bits. The controller applies a fixed
open-loop drag and is used for contact timing.

| N | Batch step wall | Aggregate env-steps/s | Aggregate sim s/wall s | 60-step run wall | Sampled process MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.518 s | 0.659 | 0.01317 | 92.96 s | 812 |
| 8 | 8.297 s | 0.964 | 0.01928 | 501.82 s | 1228 |

## Matched Reference and legacy reduced geometry

Both profiles ran the same 10-step preload and 50-step drag with identical
alternating `L100_t4` and `L150_t4` slots and seeds. All rows completed with
table contact in all 50 steps and no invalid bits.

| N | Geometry | Particles in batch | Grid dx | dt | Substeps | Batch wall | Aggregate env-steps/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Reference | 52,224 | 1.000 mm | 3.827 µs | 5,226 | 1.519 s | 0.658 |
| 8 | Reference | 520,192 | 1.000 mm | 3.827 µs | 5,226 | 6.837 s | 1.170 |
| 1 | Legacy reduced | 22,176 | 1.333 mm | 5.102 µs | 3,920 | 0.811 s | 1.233 |
| 8 | Legacy reduced | 220,608 | 1.333 mm | 5.102 µs | 3,920 | 2.427 s | 3.297 |

Legacy reduced increased throughput by 1.87× at N=1 and 2.82× at N=8.
The N=1 Reference slot and seed also appear in the full-slot row. That
independent repetition differs by under 0.05% in throughput.


## CPU numerical controls

At the stiff/light β=20 reset corner, the actual Prony guard flagged all
1,152 particles on the first rest substep at nominal CFL 1.0. The controls
at 0.5, 0.65 and 0.8 recorded zero flags in that substep. This rules out
using nominal CFL 1.0 across the declared material range.

The tiny supported strip carried its weight within 0.3% in all three
static-load arms. Raising CFL from 0.5 to 0.65 raised median tail nodal
penetration from 0.0130 to 0.0206 µm. Scaling the normal penalty fraction
to match the original stiffness recovered 0.0128 µm. These fixed-load
values diagnose the timestep dependence of the penalty law. Drag and
release have their own contact checks.

One particle per cell made a transverse checkerboard displacement mode
invisible to both interpolation and the deformation gradient at all 48
quadrature points. The same mode with two particles per axis reached
0.25 in interpolation and 500 m⁻¹ in gradient magnitude. This motivates
keeping one-particle quadrature in the experimental sweep.


## Fast and Turbo Rubber

Both opt-in profiles use nominal CFL 0.65, two particles per cell axis,
two Prony branches, consistent-mass grip projection and v2 table contact.
Fast uses 3 cells through thickness, β=10 and float64 shadows. Turbo uses
2 cells, β=5 and float32 state. The constitutive Prony state retains its
existing precision. Both update MUSL momentum only at stencil nodes
affected by the F8 particle correction, so the second global scatter is
unnecessary. The Reference task keeps its original scatter and CFL.

| N | Profile | Particles in batch | dx | dt | Substeps | Batch step | Aggregate env-steps/s | Gain vs paired Reference | Factory-to-close, 100 steps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Fast | 22,176 | 1.333 mm | 9.099 µs | 2,198 | 0.451 s | 2.219 | 3.37× | 40.81 s |
| 1 | Turbo | 6,656 | 2.000 mm | 18.248 µs | 1,096 | 0.195 s | 5.136 | 7.80× | 18.35 s |
| 3 | Fast | 77,328 | 1.333 mm | 9.099 µs | 2,198 | 0.640 s | 4.686 | 4.69× | 60.78 s |
| 3 | Turbo | 23,168 | 2.000 mm | 18.248 µs | 1,096 | 0.216 s | 13.858 | 13.87× | 20.61 s |
| 8 | Fast | 220,608 | 1.333 mm | 9.099 µs | 2,198 | 1.254 s | 6.380 | 5.45× | 124.15 s |
| 8 | Turbo | 66,048 | 2.000 mm | 18.248 µs | 1,096 | 0.288 s | 27.777 | 23.74× | 27.89 s |

The timed window is 50 synchronized drag steps after 10 preload steps.
All listed rows completed 40 more steps with the grip released and zero
invalid states. Every environment recorded a nonzero table normal force
in all 50 timed steps. Matched slots and seeds keep the workload pairing
consistent across resolutions.

The matched N=3 factory-to-close wall time for 100 steps was 290.53 s
for Reference, 60.78 s for Fast and 20.61 s for Turbo. This corresponds
to 37.2, 177.7 and 524.0 measured 100-step episodes/GPU-hour.
These runs include build, reset, rollout, diagnostics and close.

### Quality and negative results

| Profile | Max drag tip error | Max release tip error | Max drag span difference | Change in normal impulse, max | Change in shear impulse, max | Worst sampled nodal penetration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast | 0.426 mm | 0.663 mm | 0.189 mm | 0.9% | 10.2% | 0.018 mm |
| Turbo | 1.076 mm | 2.075 mm | 0.554 mm | 3.1% | 17.7% | 0.028 mm |

The quality table compares seeds 4000–4002 with matching slots at every
sample time. The grip force was zero in the release interval. The sampled
particle centres stayed above the table. Marker sag, tip sliding, force
history and material extrema are retained in the receipts. Table contact
forces and strain are provisional simulation quantities.

Nominal CFL 0.8 passed the 100-step fixed drag and release, then tripped
the actual Prony guard at step 73 (Fast) and step 36 (Turbo) on the
stiff/light seed 872. Both final CFL 0.65 profiles reran all 250 steps
with zero flags. The simple goal-seeking stress driver reached the
horizon and the frozen evaluator returned `NOT_SUCCESS` for both.
The stress test checks stability under the attempted motion.

The 64-substep eager Fast pilot attributes 16.1% of sampled CUDA
kernel time to both MUSL passes plus particle projection. Its graph
A/B at N=8 rose from 5.835 to 6.363 env-steps/s for the moderate
Fast arm and from 18.818 to 19.749 for the moderate Turbo arm.
A separate eager vector run of the same moderate Fast arm reached
5.431 env-steps/s. Higher nominal CFL, lower bulk ratio, one Prony
term, float32 and lumped grip were screened separately. The single
point per cell arm recorded zero loaded table steps in its timed
drag and remains a failed research probe. Its checkerboard null
mode also rules out promotion based on short-run speed alone.

Reference replay on CPU matched all 37 arrays across its original
and integrated code paths, including a masked reset and the final
material state. The original source digests remain auditable at
the frozen Reference commit.

## Headless CUT and TRIM

CUT uses the fixed bar and `ReferenceCommandPolicyV1`. TRIM replays
the preserved WP-17 measured blade and two pad trajectories, with seed
512 at N=1 and alternating 512/15 at N=8. Both benchmark paths run the
batched v7.2 material solver on physical GPU 1. The cut law resolves
mechanical separation through the packed material history. Headless
timings cover kinematic blade and pad inputs and material simulation.

| Task | N | Profile | Timed steps | Tool-active steps per env | Aggregate env-steps/s | Peak sampled process MiB | Full batch wall |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| CUT | 1 | Reference | 206–265 | 32 | 7.058 | 944 | 103.59 s |
| CUT | 1 | Reference | 234–293 | 60 | 6.802 | 860 | Window only |
| CUT | 8 | Reference | 234–293 | 60 | 10.426 | 1,368 | Window only |
| CUT | 1 | Coarse | 234–293 | 56 | 40.697 | 770 | 18.27 s |
| CUT | 8 | Coarse | 234–293 | 56 | 189.781 | 944 | 31.43 s |
| CUT | 1 | 4 mm grid, 0.20 ms | 234–293 | 56 | 45.293 | 770 | 17.49 s |
| CUT | 8 | 4 mm grid, 0.20 ms | 234–293 | 56 | 192.667 | 944 | 31.93 s |
| TRIM | 1 | Reference | 270–329 | 55 | 6.480 | 1,040 | 174.07 s |
| TRIM | 8 | Reference | 270–329 | 53–55 | 9.517 | 1,410 | Window only |
| TRIM | 1 | Coarse | 270–329 | 52 | 6.553 | 794 | 45.54 s |
| TRIM | 8 | Coarse | 270–329 | 50–52 | 44.928 | 1,034 | 94.00 s |
| TRIM | 1 | 4 mm grid, 0.10 ms | 270–329 | 52 | 13.587 | 868 | 51.18 s |
| TRIM | 1 | 4 mm grid, 0.20 ms | 270–329 | 52 | 26.341 | 794 | 28.39 s |
| TRIM | 8 | 4 mm grid, 0.20 ms | 270–329 | 50–52 | 131.672 | 1,066 | 58.24 s |

Reference uses 1 mm particles, a 4 mm grid and 200 material substeps at
0.1 ms per 20 ms control step. Coarse uses the numerical settings of
`research_fast_v1`, 2 mm particles, an 8 mm grid and 100 substeps at 0.2 ms,
with the existing `alias_free_v2` particle contact. The comparison measures
their combined effect. GPU samples saw desktop graphics on GPU 1 with zero
reported foreign SM load and no competing compute processes.

CUT Reference completed 700 steps and passed integrity and outcome.
Each coarse CUT episode completed 700 steps and passed integrity. All
nine runs produced two 72 g components with full mass retained. The
coarse interface statistic was 3.0 mm, equal to the ideal-partition
statistic at that resolution. The frozen 2.0 mm location check failed.
The N=8 contact window was 18.20× faster in aggregate, and the N=1 full
episode took 5.67× less wall time. This is a coarse physical CUT path
for research where millimetre-scale location error is acceptable.

Reference TRIM completed 1,200 steps and passed integrity and outcome
for seed 512. The target separated as a 0.498 g chip. The 153.6 g
substrate was preserved, with a 1.5 mm reference-interface statistic.
Coarse TRIM passed material integrity and retained the substrate. Every
target stayed attached, yielding a frozen outcome of `FAIL` at N=1 and
in all eight runs at N=8.

The independent terminal-state review found all 41 seed-512 target
particles on the positive cut digit. Six 8 mm reference nodes still
linked 11 of those particles directly to unassigned, code-zero particles
below the trim plane. Rebuilding the stencils at 4 mm on the same
particles and labels recovered two components. Rebuilding the
Reference stencils at 8 mm yielded one component. This CPU diagnostic
isolates stencil connectivity at fixed labels. The live grid sweep
exercised cut-front propagation and contact on the same blade replay.

The opt-in `research_fast_4mm_v1` uses 2 mm particles on the 4 mm grid,
with 100 substeps at 0.20 ms. Both N=1 episodes passed the frozen
integrity and outcome checks. CUT produced two 72 g pieces and a
1.0 mm interface statistic. TRIM removed 100% of the target, with a
0.492 g chip, all 153.6 g of substrate retained, and a 1.0 mm
reference-interface statistic. The 0.10 ms TRIM control reached the same
topology and interface statistic. At 0.20 ms, full N=1 runs were
5.92× faster for CUT and 6.13× for TRIM against their original-resolution
runs. Keeping 1 mm particles and changing the step to 0.20 ms also
passed TRIM integrity and outcome. Its contact-window rate was
5.203 env-steps/s.

The named profile passed all eight complete episodes for each task.
Every CUT produced two 72 g pieces with a 1.0 mm interface statistic.
Every TRIM separated all targets, including both chips in seed 15.
Measured substrate retention was 100%, with no detached below-plane
mass and the 1.0 mm reference-interface statistic. Window throughput
at N=8 was 18.48× Reference for CUT and 13.84× for TRIM. Both
headless N=8 Reference rows simulated only the window and its prefix.

## Native CUT and TRIM

The live runs used Isaac Sim 6.2, PhysX for robot, gripper and tool
dynamics, and CUDA Graphs for the coupled material. The GPU namespace
exposed only physical GPU 1. Timing includes a cold per-run cache.
The CUT controller uses measured edge pose. It records rig v5 views
and an overview at 5 Hz. TRIM runs the frozen WP-17 C3 controller with
rig14 Global RGB and wrist-0 RGB-D at 5 Hz.

The 1,200-action Native TRIM Reference completed in 375.52 s including
67.30 s to construct and reset and 11.67 s to finalize and score.
The controller took 296.51 s. During its control steps the material
CUDA Graph took 153.63 s, camera render and readback 29.60 s, and
IK 20.97 s. Stage timers are inclusive. The independent evaluator
passed, the whole target separated, and the 0-to-24 s capture timeline
has 121 frames with verified MP4 decode and timestamps.

Native CUT Reference reached 700 task steps at 14.00 s, followed by
40 HOME steps through 14.80 s. Task integrity, task outcome and HOME
all passed. The recorded task-end state had two 72 g pieces and a
1.5 mm plane statistic. The 291.99 s run includes plan preflight,
startup, the task, HOME and raw recording. The process returned after
293.33 s. Post-run MP4 verification covered all 75 original overview
captures through HOME.

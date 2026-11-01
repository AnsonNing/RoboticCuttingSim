# WP-14 compiled-planner refinement and scoped adoption

IN_PROGRESS, 2026-11-01 Asia/Seoul. No adoption or publication completion is
claimed yet. General default: v5 / Law 8 / particle_surface_v1. Camera rig v5,
WP-12/WP-13 protocols, records and solver sources are preserved. Materials
remain provisional and uncalibrated.

The core protocol was committed at `cd02821` before outcomes. It fixes seven
matched configurations (14 episodes) using the corrected compiled planner.
Actual commands are independently replayed from compiled controller settings
and recorded deployable kinematic proprioception, including every pose,
phase, cut id, speed and enable flag. A matching task id alone is insufficient.

| Completed pair | v5 outcome | v7 outcome | P95 consistency gain | Local roughness reduction |
| --- | --- | --- | --- | --- |
| C00 standard CUT | PASS, two 72 g parts | PASS, two 72 g parts | 6.05x | 85.94% |
| C01 short/firm CUT | PASS, two parts | PASS, two parts | 8.49x | 78.48% |
| T00 dome TRIM | PASS, detached chip | FAIL, one body | 6.83x | 85.47% |
| T01 asymmetric TRIM | PASS | FAIL, one body | pending compact audit | pending |
| T02 two-bump TRIM | PASS | FAIL, one body | pending compact audit | pending |
| M00 four slices | running | NOT_RUN | NOT_RUN | NOT_RUN |
| P01 distinct shallower Partial | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

The first three pairs have identical actual commands and PASS compiled replay.
TRIM quality gains do not establish separation. T00 candidate fixture overlap
is 0.467 mm versus baseline 0.055 mm. The original WP-13 shorter/firm failure
remains historical evidence under its actual legacy path; this new corrected
C01 result does not replace it.

A fresh read-only cuda:0 T00 trace reproduces the one-body failure. All 216
selected code-0 bridges record a finite leading-plane crossing but zero
particle cutting work and zero engaged-bevel substeps. Engagement occurs only
after they leave the 3.383 mm bevel. These particles are upstream base material,
not missing chip mass. Their local vertical-neighbour links do not straddle
the cut plane; no unsupported crossing history is inferred from that trace.

A second targeted trace instead selects actual positive-side MPM support
neighbours sharing a nonzero-weight/gradient node. All 216 selected moving
interfaces cross the finite edge; 152 do so while the stroke is engaged.
This is an actual new substep observation, not relabeling a final snapshot.
Both traces are diagnostic repeats on GPU 0; the original paired cohort
continues untouched on GPU 1. Raw:
`artifacts/wp14/{trim_trace_001,trim_support_trace_001}/front_trace.json`.

The finite `tlmpm_v7_1` experiment shares v7 x/F/contact and completes a digit
only after a recorded finite support-interface sweep, within that interface's
physical bevel residence, with actual directional knife contact work at its
other endpoint. It uses reference support adjacency, no target plane or task
family. The original v7 and planned CUT profile remain separate. CPU 15 PASS;
first expired-event test failure retained before correction. Full candidate
GPU task outcome is NOT_RUN.

Six v7 physical negative controls PASS; six command replay/trace checks PASS.
Two unchanged surface profiles were measured on six saved CUT/TRIM states,
including a bounded 512-triangle intersection proxy. No renderer edits or
smoothing follow these measurements. Baseline TRIM cut-aware reports sampled
intersections; a zero sampled count does not prove a defect-free full mesh.

Four held-out CUT pairs and two Native CUT pairs were predeclared at `a0b31c1`.
Their proposed support envelope is fixed in
[heldout_v1](../../configs/wp14/heldout_v1.json); outcomes are NOT_RUN.
One half-dt CUT pair was separately frozen for sensitivity, not convergence.
Native/media, complete stable-source CPU CI, final read-only review, PR/main
merge and exact-commit remote CI/playback remain NOT_RUN. Completion markers
are written only after all release gates actually pass.

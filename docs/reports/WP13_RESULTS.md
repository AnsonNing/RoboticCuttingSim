# WP-13 integrated cutting experiment

Status: IN_PROGRESS, 2026-11-01 KST. Baseline `11895016b34ed821be2bdecbc5351667fb62d47d`.
The production default remains TLMPM v5, Law 8, `particle_surface_v1`.
Material parameters are provisional and uncalibrated.

## A — Baseline and decisive diagnosis

The WP-12 baseline and failures are retained unchanged. Its v6 flat-board CUT
left twelve code-0 particles at the bottom side corners and one mechanical
component; v5 produced two. Existing baseline metrics, source hashes, installed
versions, exact diagnostic command and new observations are in
[phase_a_001.json](../evidence/wp13/phase_a_001.json).

The new read-only substep trace repeats the failure (one component). Eight
traced outer-corner particles remain 8–32 micrometres **ahead** of the advancing
edge even at their closest approach, while both the stroke and segment gates
are active. The knife therefore did not cross these particles. This is a
support/contact and motion problem, not a missing permissive label tolerance.
The diagnostic samples the two outer y faces; the twelve-particle historical
bridge includes four particles one y layer inward. GPU repeat differences are
retained rather than treated as bitwise reproduction.

Raw trace: `artifacts/wp13/front_trace_001/front_trace.json`, 700 control steps,
approximately 204 s episode wall, seed 42, cuda:1. No historical output was
modified. The trace observes every substep through a CUDA graph; it writes no
physical state. The original recorded scene/goal, planner and evaluator run
through the existing WP-12 driver.

## B/C — Candidate under development

`tlmpm_v7` is opt-in. It integrates x and F from the same MUSL field, evaluates
blade and board contact with a velocity-level normal constraint and a fixed
0.2 ms penetration-relaxation time, and assigns persistent digits only when
the finite advancing front crosses the moving particle layer in that substep.
The edge release region is defined in metres; flank reactions remain active.
The shared blade SDF, physical interaction interval and work-engaged directional
stroke feed contact and front assignment. The existing read-only cut-event
recorder supplies link intersections and Hermite provenance downstream.

This is a numerical cutting experiment, not a calibrated fracture model. It
does not add stress-driven damage, remove particles, smooth meshes or promote
the rejected v6. Physical outcomes, contact stability, quality and generalization
are NOT_RUN until their actual results are recorded below.

One initial CPU construction failed because the base constructor calls
`set_trajectory` before the candidate allocates its previous-pose buffer. That
failure is retained in the runner events; the corrected CPU compile is in
`artifacts/wp13/v7_cpu_compile_002.log`. It does not establish cutting success.

## D–F — Pending

Surface comparisons, precommitted held-out matrix, Native runs, independent
review, animated gallery, full CPU CI and remote main publication remain
NOT_RUN. DONE has not been written.

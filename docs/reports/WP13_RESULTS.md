# WP-13 integrated cutting experiment

Status: candidate **REJECTED FOR PROMOTION**; engineering publication IN_PROGRESS,
2026-11-01 KST. Baseline `11895016b34ed821be2bdecbc5351667fb62d47d`.
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

## B/C — Experimental candidate

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
are assessed separately below.

The first complete candidate CUT failed (one component, 13,080 labels): the
0.5 mm work region lost directional engagement halfway down the bar. Its
[receipt](../evidence/wp13/dev_cut_001.json) and exact matching source are retained.
The single focused alternative uses the actual 3.383 mm bevel for cutting work,
retains flank reactions, and stores a finite sweep until actual work confirms
engagement while that layer is still at the bevel. It does not label unswept
layers. This alternative CUT passed the existing evaluator: two 72 g pieces,
near-cut consistency P99 0.311 spacings, local face roughness about 0.043 mm RMS,
fixture overlap 0.031 mm and blade overlap 0.125 mm. These are development
measurements, not matched A/B acceptance or generalization.
[Alternative receipt](../evidence/wp13/dev_cut_alternative_001.json).

One initial CPU construction failed because the base constructor calls
`set_trajectory` before the candidate allocates its previous-pose buffer. That
failure is retained in the runner events; the corrected CPU compile is in
`artifacts/wp13/v7_cpu_compile_002.log`. It does not establish cutting success.

CPU contact/front/rigid-translation controls: 13 passed, including six physical
negative controls (no contact, side push, flat press, blunt, stop, withdrawal).
One initial rigid-translation test used the documented zero-mass node-plane
lattice defect; using a half-grid lattice verifies the intended affine identity
without changing its tolerance. A first test command named a nonexistent scene
test; it collected nothing and remains an unsuccessful invocation.

The recorded C00 CUT pair succeeds in both arms (legacy goal-derived planner;
see the execution-integrity deviation below): v7 near-cut consistency P95
and P99 improve by 6.06× and 5.73×, and midpoint-locus patch roughness falls
85.96%. These are one-pair results, not generalization. The first v7 run took
1.74× the baseline wall because its diagnostics performed global atomic writes
even for zero impulses. Skipping those zero writes preserves CPU physical state,
reactions and diagnostic sums bit for bit over 300 substeps; its original slow
run and source are retained rather than rerun for timing.

A second instrumentation defect was found in cumulative float32 impulse sums:
the fixture sum can stagnate at 1 N·s while tiny increments continue. Those new
cumulative partitions in the early CUT/PARTIAL records are **INVALID** and are
not used as force evidence. Existing per-control reaction/work/penetration
records are separate and remain available. Later diagnostics sum in float64;
CPU physical state and reactions remain bitwise identical in the focused check.
No physical update formula changes; GPU bitwise identity is not claimed.
The initial incorrect diagnostic
cast edit failed CPU compilation and was preserved, then corrected before the
next candidate episode. No physical parameter or cutting criterion was tuned.

## E — Frozen representative cohort

**Decisive rejection:** the primary frozen T00 TRIM succeeds on v5 (two
components), but v7 remains one component and fails
`selected_protrusions_removed`. Better x/F consistency and lower local locus
roughness do not establish chip detachment. The one focused physical alternative
has already been exercised; no further solver tuning follows this failure.
The candidate stays explicitly experimental and the production default remains
v5. The remaining predeclared matrix and Native episodes characterize its limits
and support the authorized rejection-publication path.

**Native is separate:** fresh Native v7 T00 TRIM passes its task evaluator and
returns HOME, with actual `tlmpm_v7` selection verified in its material receipt.
This does not erase the failed ideal-tracking episode. The real measured tool
path, coupling and GPU nondeterminism differ; their causal contributions have
not been isolated. Both outcomes and denominators remain in the report.

Read-only final-stencil diagnosis finds **216 direct code-0 bridges** on v7
TRIM versus zero on v5. Although 1,044 particles have the positive-side digit,
the above-plane reference mass removed is exactly zero because the material
remains one mechanical body. Per-substep rejection reasons were not recorded
for those unassigned particles, so a specific missed-crossing history is not
invented retrospectively. [Bridge evidence](../evidence/wp13/trim_bridge_diagnosis_001.json).

The separate 2 mm small-block recontact control also fails its initial severing
precondition (retained CPU failure / expected unsupported result, not a pass).
One necessary 1 mm GPU control passes: two components, unchanged labels/bonds
through disarmed withdrawal and push, actual cross-component particle distance
0.689 mm versus the 0.9 mm contact diameter, measured push contact, and 144 finite
Hermite events without sign-inconsistent records. This is a kinematic control,
not Native or real-material validation.

[focused_v1](../../configs/wp13/focused_v1.json) fixes 16 configurations × two
solver arms = 32 kinematic episodes and three configurations × two arms = six
Native episodes. It includes aspect ratios, Gaussian/dome bumps, one/two bumps,
partial depths, two/four cuts, explicit ±20° slotted cuts, yaw, stiffness/yield/
friction variations, half timestep and one 0.75 mm spacing spot check. Seeds
are recorded; deterministic shape configurations do not acquire random shape
diversity merely by changing their seeds. No outcome-based redraw is allowed.
The rotated-board configuration initially failed compile-time footprint checks;
its board centre was moved 25 mm inward before any formal physics outcome.
The two-bump draft needed two retained digits, also corrected before outcomes.

## D — Unchanged surface profiles

Both unchanged v1 profiles were extracted on 15 saved states across CUT,
PARTIAL, TRIM and four-slice episodes, including during-cut frames. The baseline
C00 during-state has no recorded F and is NOT_RUN; completed physics was not
repeated. Midpoint-fallback provenance is explicit; no Hermite data is invented.
No renderer change or surface promotion follows a failed primary physical TRIM.
[Surface measurements](../evidence/wp13/surfaces_001.json).

## Native integration and media

Six fresh paired Native episodes completed: **6/6 TASK_SUCCESS, 6/6 HOME PASS**.
CUT and four slices produce two and five components respectively. T00 removes
97.5096% of above-plane mass with v5 and 100% with v7. These are selected
predeclared demonstration scenes, not random generalization. Legacy v1 precision
metrics remain frozen and separate from task outcome; a successful divide task
can still have shifted cut locations. Native F was not recorded, so Native x/F
consistency is NOT_RUN. [Complete compact receipts](../evidence/wp13/native_001.json).

Six initial setup attempts failed before physics due to the isolated worktree's
missing asset-cache path. Existing assets were linked read-only; a fresh full
547-sample Native geometric preflight passed before the six new actual episodes.
Original failures, incomplete analysis metadata and unsuccessful invocations are
retained under `artifacts/wp13`; no outcome was replaced.

The gallery uses genuine recorded RGB only: six full clips, a matched-time CUT
comparison and five looping GIF previews. Capture timestamps and hashes are in
[media provenance](../evidence/wp13/media_001.json). Full videos retain all frames
at nominal 5 fps; presentation speed and crops are labelled. Local full decode,
links and 178 protected Git blobs PASS. Rendered remote main playback, exact-main
CI, independent review and final matrix aggregation are still pending.


## Independent review and execution-integrity deviation

The independent sequential read-only xhigh review of `82fabb0` found a real
P2 driver bug: the reused historical kinematic driver recomputes planner options
from the goal and ignores the compiled controller overrides. It executes P01
with `overtravel_m=-0.010` rather than its requested `-0.012`; P01 and P00
record identical paths. P01 is **duplicate-depth evidence**, not a successful
new partial-depth setting. Broader audit also finds dropped controller-profile
and action-limit overrides. The manifests preserve actual values; compiled task
identity alone is insufficient proof of execution.

All paired arms use the same actual legacy options. Their observed outcomes and
quality differences are real under those paths, but the batch's requested-planner
integrity gate FAILS. Native launches explicitly pass the compiled options and
are separate. The running frozen batch retains its original driver until the dt
pairs finish, so contact comparisons do not mix control paths. Completed episodes
are not repeated. The driver repair and regression check follow batch completion;
no corrected-depth outcome is claimed. [Original review](../evidence/wp13/review_001.json).

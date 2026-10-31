# WP-13 integrated cutting experiment

Status: candidate **REJECTED FOR PROMOTION**; research payload complete; engineering release gates below,
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

## B — Experimental candidate

`tlmpm_v7` is opt-in. It integrates x and F from the same MUSL field, evaluates
blade and board contact with a velocity-level normal constraint and a fixed
0.2 ms penetration-relaxation time, and assigns persistent digits only when
the finite advancing front crosses the moving particle layer in that substep.
The edge release region is defined in metres; flank reactions remain active.
The shared blade SDF, physical interaction interval and work-engaged directional
stroke feed contact and front assignment. The existing read-only cut-event
recorder supplies link intersections and Hermite provenance downstream.

The assignment remains particle-layer based with the inherited 6-grid-cell
side-code support band and blade-wide stroke persistence. It is **not** a
complete per-reference-link intersection/engagement solve. A real leading-plane
crossing is stored, then accepted while that layer remains in the physical
bevel and the stroke has work engagement. The failed TRIM demonstrates that this
still does not guarantee closure of all mechanical interfaces. Handle/obstacle
and separated-particle contacts retain the baseline law; velocity projection
replaces only the candidate blade/support paths. This is partial integration.

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

## C — Contact sensitivity, not convergence

The unchanged physical geometry CUT pair at 0.1/0.05 ms passes in both arms.
Recorded reaction is averaged over a 20 ms control interval; the peak below is
not an instantaneous substep force.

| CUT solver / dt (ms) | Peak reaction norm (N) | Mean norm on contact controls (N) | Blade overlap max (mm) | Support overlap max (mm) | Tool work (mJ) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v5 / 0.10 | 0.0446 | 0.0102 | 0.4831 | 0.0810 | 0.2547 |
| v5 / 0.05 | 0.1216 | 0.0336 | 0.1213 | 0.0340 | 0.7792 |
| v7 / 0.10 | 0.1341 | 0.0627 | 0.1243 | 0.0308 | 1.3757 |
| v7 / 0.05 | 0.1675 | 0.0831 | 0.0748 | 0.0212 | 1.8030 |

On this pair v7 peak force changes about 25% and work 31%, versus about 173%
and 206% on v5. This is **less sensitive in this CUT spot check**, not converged
or calibrated. Only two dt values were measured, and final-state/force agreement
is not a general convergence proof. The nominal fixed blade release geometry is
0.5 mm depth / 0.25 mm half-thickness; particle contact radius still follows
spacing, which remains a discretization dependence.

At 0.1 ms, TRIM v7's support overlap is 0.4578 mm versus v5's 0.0554 mm,
although blade overlap falls from 0.6038 to 0.1515 mm. The candidate therefore
does not uniformly improve contact. Later float64 receipts separately report
normal/tangent impulse sums, bevel/flank partitions, consumed cutting work,
fixture impulse and pretransfer impulse-energy expressions. Those expressions
exclude grid transfer and material contacts: **total-system energy balance is
NOT_RUN**. Fixture reaction for v5 was not recorded; support overlap is not a
substitute for that missing force measurement.


TRIM at 0.05 ms still fails to detach on v7, while v5 passes. Its reaction peak,
contact-control mean, blade/support overlap and work are respectively:
v5 0.1637 N / 0.0395 N / 0.4391 mm / 0.0151 mm / 0.6804 mJ;
v7 0.1046 N / 0.0361 N / 0.1194 mm / 0.3640 mm / 2.0687 mJ.
v7's corresponding 0.1 ms values are 0.0890 N / 0.0321 N / 0.1515 mm /
0.4578 mm / 1.8397 mJ. The persistent topology failure and worse support overlap
remain decisive limitations even where reaction varies less with dt.

## D — Unchanged surface profiles

Both unchanged v1 profiles were extracted on 15 saved states across CUT,
PARTIAL, TRIM and four-slice episodes, including during-cut frames. The baseline
C00 during-state has no recorded F and is NOT_RUN; completed physics was not
repeated. Midpoint-fallback provenance is explicit; no Hermite data is invented.
No renderer change or surface promotion follows a failed primary physical TRIM.
[Surface measurements](../evidence/wp13/surfaces_001.json).

| Final state / unchanged profile | v5 mesh roughness RMS (mm) | v7 (mm) | v5/v7 normal P95 (degrees) | v5/v7 locus coverage |
| --- | ---: | ---: | ---: | ---: |
| CUT / particle surface | 0.1556 | 0.0123 | 26.96 / 2.32 | 0.953 / 1.000 |
| CUT / cut-aware v1 | 0.2523 | 0.0312 | 29.65 / 5.57 | 1.000 / 1.000 |
| Four slices / particle surface | 0.1390 | 0.0131 | 29.82 / 2.55 | 0.815 / 1.000 |
| Four slices / cut-aware v1 | 0.3053 | 0.0333 | 38.17 / 5.58 | 1.000 / 1.000 |

These mesh metrics use the existing core-region and 2 mm local-patch rules on
the actual saved x/F/codes, not distance to an ideal flat cut. Straight CUT's
rigid-aligned actual face-locus whole-plane P95 falls from about 1.234 mm to
0.255 mm, RMS 0.683 to 0.125 mm. Bias is deformation relative to its own reference
midpoints, not calibrated cut-position accuracy. A curled TRIM shaving is not
penalized for failing a global plane fit.

The partial slit retains one body. Its default-profile locus coverage falls
from 0.537 to 0.430 despite lower local roughness; cut-aware v1 reaches 1.000.
TRIM's v7 mesh metrics improve locally while the chip remains attached, so they
do not pass physical correctness. Basic mesh reports include degenerates
(e.g. two in each CUT particle surface, one in each four-slice surface), even
where topology is watertight and non-manifold edges are zero. Self intersections
are NOT_RUN; no claim of fully valid or universally spike-free meshes follows.
During-state timestamps differ across arms; those frames show slit evolution,
not matched-time quantitative A/B. No surface v2 or smoothing was introduced.

## E — Frozen representative cohort

**Decisive rejection:** the primary frozen T00 TRIM succeeds on v5 (two
components), but v7 remains one component and fails
`selected_protrusions_removed`. Better x/F consistency and lower local locus
roughness do not establish chip detachment. The one focused physical alternative
has already been exercised; no further solver tuning follows this failure.
The candidate stays explicitly experimental and the production default remains
v5. The completed predeclared matrix and Native episodes characterize its limits
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

The final recorded batch is **32/32 completed**, 16 paired configurations.
Observed task/control-rule results are **v5 14/16, v7 9/16**. These counts include
two PARTIAL controls; P01 duplicates P00's executed depth. All 16 pairs have
identical recorded time/phase/cut-id/edge-pose paths across solver arms, but
**0/32 match all compiled planner settings**. This is descriptive paired evidence
under actual legacy paths, not a success rate for the requested protocol.
[All completed rows, source hashes, commands and deviations](../evidence/wp13/focused_001.json).

| Recorded case | v5 | v7 | Boundary |
| --- | --- | --- | --- |
| C00 standard CUT | PASS | PASS | Two pieces |
| C01 shorter/firmer CUT | PASS | FAIL | v7 remains one body |
| C02 larger/yawed/low-friction CUT | PASS | PASS | Held-out geometry/material |
| C03 offset CUT | FAIL | FAIL | Both remain one body |
| T00 dome TRIM | PASS | FAIL | v7 chip attached |
| T01 asymmetric Gaussian/clay TRIM | PASS | FAIL | v7 chip attached |
| T02 two-bump TRIM | PASS | FAIL | v5 four components; v7 one |
| P00 PARTIAL | Control PASS | Control PASS | One body with slit and intact ligament |
| P01 requested shallower PARTIAL | DUPLICATE | DUPLICATE | Actual control passes; requested depth NOT_RUN |
| M00 four slices | PASS | PASS | Five pieces; placement quality separate |
| M01 two cuts | PASS | PASS | Three pieces |
| O00 plus-20-degree groove | PASS | PASS | Explicit 8 mm slotted board |
| O01 minus-20-degree groove | PASS | PASS | Explicit 8 mm slotted board |
| D00 CUT at 0.05 ms | PASS | PASS | Two pieces |
| D01 TRIM at 0.05 ms | PASS | FAIL | v7 remains one body |
| R00 CUT at 0.75 mm | FAIL | FAIL | Both remain one body |

The original sever evaluator intentionally fails both PARTIAL cases; the separate
predeclared partial-control rule checks the connected ligament and real kerf.
Its observed passes do not validate P01's missed requested depth. No failures or
protocol deviations are dropped from the denominator.

For the seven actual cutting pairs that succeed in both arms, near-cut x/F P95
improves by 4.25–6.95× and local locus roughness falls 38.8–86.4%. Examples beyond
C00 include C02 (5.89× / 86.4%), four slices (6.53× / 85.9%) and minus-20-degree
slot (6.21× / 38.8%). The same metrics also improve on several failed cuts;
quality gain does not imply successful separation. All final states are finite;
minimum final det(F) is 0.869, sampled minimum 0.757 (every ten controls, not all
substeps). Maximum relative mass bookkeeping error is 1.22e-12; no particles are
deleted. No sparse-slot overflow is reported.

Median paired episode-wall ratio v7/v5 is 1.007; maximum is 1.743 for the preserved
first diagnostic-heavy CUT. Reported process GPU memory peaks at 1,254 MiB. These
are instrumented, single-episode measurements, not universal performance bounds.
The one 0.75 mm spot check fails in both arms and samples about 145.57 g rather
than the 1 mm lattice's 144 g; particle radius and discrete mass change with
sampling. TRIM at 0.75 mm is NOT_RUN. No resolution-convergence claim follows.

## Native integration and media

Six fresh paired Native episodes completed: **6/6 TASK_SUCCESS, 6/6 HOME PASS**.
CUT and four slices produce two and five components respectively. T00 removes
97.5096% of above-plane mass with v5 and 100% with v7. These are selected
predeclared demonstration scenes, not random generalization. Legacy v1 precision
metrics remain frozen and separate from task outcome; a successful divide task
can still have shifted cut locations. Native F was not recorded, so Native x/F
consistency is NOT_RUN. [Complete compact receipts](../evidence/wp13/native_001.json).

Native CUT position-only midpoint geometry also improves: whole-face plane P95
is about 1.210 mm on v5 versus 0.235 mm on v7; local locus roughness is about
0.296 mm versus 0.037 mm. This uses translation-only midpoint reconstruction
because Native F is absent, and is distinct from the F-aware kinematic metric.
Four-slice legacy v1 precision remains 5.0 mm in both arms even though the
independent task evaluator passes: open-loop placement/sliding remains visible.
No frozen v1 threshold or result has been changed.

| Native case | Process wall v5/v7 (s) | Mean control wall v5/v7 (ms) | Mean material wall v5/v7 (ms) |
| --- | ---: | ---: | ---: |
| CUT | 386.67 / 387.90 | 405.11 / 404.62 | 270.08 / 268.81 |
| TRIM | 354.16 / 354.06 | 427.07 / 428.69 | 275.06 / 271.64 |
| Four slices | 993.90 / 968.54 | 428.16 / 414.37 | 268.92 / 268.69 |

These include recording/instrumentation overhead and are not real-time or pure
kernel benchmarks. Native control-wall P95 is roughly 614–634 ms. Kinematic
run receipts also record throughput, milliseconds per 20 ms control interval
and process GPU memory; offline surface extraction timing is separate. No broad
speedup claim is based on these single paired episodes.

Six initial setup attempts failed before physics due to the isolated worktree's
missing asset-cache path. Existing assets were linked read-only; a fresh full
547-sample Native geometric preflight passed before the six new actual episodes.
Original failures, incomplete analysis metadata and unsuccessful invocations are
retained under `artifacts/wp13`; no outcome was replaced.

The gallery uses genuine recorded RGB only: six full clips, a matched-time CUT
comparison and five looping GIF previews. Capture timestamps and hashes are in
[media provenance](../evidence/wp13/media_001.json). Full videos retain all frames
at nominal 5 fps; presentation speed and crops are labelled. Local full decode,
links and 178 protected Git blobs pass. Exact-head CI is required before merge;
exact-main CI and rendered GitHub animation/full playback are verified after
merge, with receipts recorded in the PR. Filesystem presence alone is insufficient.


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
are separate. The entire frozen batch retained its original driver until all 32 episodes
finished, so dt comparisons do not mix control paths. Completed episodes were
not repeated. The corrected driver now consumes compiled options by default;
two integration regression tests inspect the real historical driver's manifest
and action limits before intentionally stopping before physics. They pass and
verify restoration of scoped overrides. `--planner-source legacy_goal_v1`
explicitly replays the recorded deviation. No corrected-depth physical outcome
is claimed; that requested depth remains NOT_RUN. [Original review](../evidence/wp13/review_001.json);
[follow-up: no blocking findings for rejection publication](../evidence/wp13/review_002.json).


## Reproducibility and scope

Base: Python 3.12.12, Warp 1.17.0, Newton 1.6.1, NumPy 2.5.4,
SciPy 1.18.1. Native: Isaac Sim 6.2.0.0, NumPy 2.3.1, SciPy 1.17.0,
Torch 2.11.0+cu128; the same Python/Warp/Newton. Hardware: two RTX 5090
32 GB, driver 570.211.01. Kinematic pairs use cuda:1 except the reserved
0.75 mm and minus-20-degree pairs on cuda:0; each pair shares its device.
Native episodes run one at a time on GPU 0. No driver, package environment,
training run, other worktree or historical experiment is modified.

The first v7 CUT source predates diagnostic optimization. Later changes remove
zero diagnostic atomics and accumulate diagnostics in float64; physical formulae
remain fixed after the one focused alternative. CPU neutrality receipts establish
exact physical-state/reaction equality for those instrumentation changes, not
GPU bitwise identity. Each actual run manifest and source hash takes precedence
over the long-lived queue launch commit. Large states remain under ignored
`artifacts/wp13`; only compact receipts and real encoded media are published.

Scientific limits remain explicit: no real-material calibration, no converged
force prediction, no stress-driven fracture, no arbitrary solid-board oblique
cutting, no general Native robustness rate, and no accepted replacement for v5.
Vision-only observations still exclude force and oracle material state. Camera
v5, goals, frozen v1 evaluation and the prior v6 rejection remain unchanged.


## Verification and release gates

The full collected local CPU suite recorded 1647 passed, 12 skipped/NOT_RUN and
five expected failures in 1912 s. Collection began at `c82b07c` and the workspace
evolved, so this is not an exact-final-head claim. After the driver fix, focused
checks record **52 passed, one known expected failure** in 26.54 s. The latter is
the preserved 2 mm severing-precondition failure, not a successful recontact test.
[Verification receipts](../evidence/wp13/local_verification_002.json).

The scientific acceptance result is **REJECTED**: primary CUT/TRIM correctness
is not retained across the recorded paired set, and requested-planner integrity
fails. Contact/material validity remains PARTIAL/uncalibrated. Engineering
publication keeps v5/default surface and frozen evidence, supplies functioning
Native media and repairs the future execution path. Independent review and
protected-file/media checks precede the PR; green exact-commit CPU CI and real
remote main playback are mandatory release checks, recorded in the PR and the
local runner's post-merge receipt. They do not promote the scientific candidate.

To reproduce a recorded legacy-path CUT in a **new** output directory:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/run_wp13.py --case configs/wp13/cases/C00_cut.json \
  --solver tlmpm_v7 --planner-source legacy_goal_v1 --device cuda:1 \
  --out artifacts/wp13/my_legacy_cut_001
```

Omit the legacy flag for the corrected compiled-planner execution; its outcomes
are not substituted for this batch. Native commands and actual versions, seeds,
source/result hashes and timing are preserved in the compact receipts. Rebuild
media only into fresh staging/output paths using `scripts/encode_wp13_media.py`.

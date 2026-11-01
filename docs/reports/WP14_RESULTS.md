# WP-14 compiled-planner refinement and scoped CUT adoption

Scoped CUT adoption selected, 2026-11-02 Asia/Seoul; final independent review
ACCEPT after fixes. Publication requires the exact-main verification below. The proposed broad envelope failed; the narrower
explicit CUT configuration is qualified in the measured instances. General defaults remain v5 /
Law 8 / particle_surface_v1. Materials are provisional and uncalibrated.
Camera rig v5, frozen evaluation v1 and all WP-12/WP-13 evidence are unchanged.

## Execution integrity and all paired outcomes

The core was committed at `cd02821` before outcomes: seven matched configurations,
14 episodes, `planner_source=compiled`. Every recorded kinematic command is
independently replayed from the compiled controller and prior recorded
proprioception. All 14 PASS; all seven paired paths are identical. Scene, goal,
material seed, effective controller, dt and evaluator match between arms.
[Complete core](../evidence/wp14/core_001.json),
[resolved materials and per-phase angles/speeds](../evidence/wp14/core_material_angles_001.json).

| Pair | v5 | v7 | Near-cut x/F P95 gain | Local physical-locus roughness reduction |
| --- | --- | --- | ---: | ---: |
| C00 standard CUT | PASS, two parts | PASS, two parts | 6.05x | 85.94% |
| C01 short/firm CUT | PASS, two parts | PASS, two parts | 8.49x | 78.48% |
| T00 dome TRIM | PASS | FAIL, attached | 6.83x | 85.47% |
| T01 asymmetric TRIM | PASS | FAIL, attached | 7.77x | 87.10% |
| T02 two-bump TRIM | PASS | FAIL, attached | 5.78x | 87.99% |
| M00 four slices | PASS, five parts | PASS, five parts | 6.47x | 85.93% |
| P01 shallower Partial | Control PASS | Control PASS | 2.44x | 81.30% |
| H00 yaw/offset reference CUT | PASS | PASS | 5.60x | 86.06% |
| H01 yaw/offset firm CUT | PASS | PASS | 8.21x | 78.00% |
| H02 +6 mm CUT | FAIL location | FAIL location | 8.77x | 87.64% |
| H03 reverse/-6 mm CUT | FAIL location | FAIL location | 15.43x | 78.62% |
| D00 same CUT, half dt | PASS | PASS | see receipt | 83.15% |

There are **24/24 completed formal kinematic episodes**: 14 core, eight held-out,
two dt. Quality is measured even on failures; it does not certify separation or
task success. Multi-cut has one tested pattern and remains Experimental.
Partial retains a real slit and one body; its full-sever evaluator correctly
FAILS. Its separate frozen partial control passes. Measured armed minimum world
Z is 22.200 mm versus historical 20.401 mm, **1.799 mm genuinely shallower**;
labels start at reference Z 12.5 mm. No duplicate-depth claim is reused.
[Depth audit](../evidence/wp14/partial_depth_001.json).

## Held-out failure and restricted scope

Four new pairs and Native C00/H01 pairs were committed at `a0b31c1` before any
held-out outcome. The original held-out task result is **2/4 per arm**, with
**4/4 true two-part severing per arm**. H02/H03 both retain their failed frozen
2 mm location checks. Their actual **and ideal-geometric** reference midpoint
statistics are 2.5 mm: this is a discretization/evaluator limitation, not evidence
that the real knife missed its commanded plane. No threshold or result changes.
[All held-out rows](../evidence/wp14/heldout_001.json).

The proposed ±6 mm envelope therefore FAILS. The profile is restricted after
these outcomes to +x blade direction, ±5 degree workpiece yaw, -4/0/+4 mm local
cut locations and 8 mm bar-length increments. Size bounds describe
120–160 x 24–32 x 16–20 mm plain bars on the declared solid flat board;
materials are the frozen reference recipe or `wp12_firm_v1` (50/70 kPa,
1/3 kPa yield, friction 0.3). Particle/grid spacing is 1/4 mm, dt 0.1 ms,
controller path_v2, cut speed 10 mm/s. Board overrides are constrained.

Four measured successful instances are C00, C01, H00, H01. These finite results
do **not** independently confirm a new envelope after narrowing, establish a
Cartesian-product guarantee, or supply a random success rate. In-between
conditions need their own evaluation. Failed rows remain in the original
held-out denominator; no redraw or success-only generalization is used.
[Original proposal](../../configs/wp14/heldout_v1.json),
[explicit restriction and frozen ideal-statistic evidence](../evidence/wp14/support_scope_001.json).

## Finite front experiment and retained TRIM failure

Two fresh read-only cuda:0 traces reproduce corrected-path T00 failure; the
formal paired cohort stays on cuda:1. All 216 selected code-0 bridges record a
finite particle leading-plane crossing, but zero particle cutting work and
zero engaged-bevel substeps. Engagement arrives after their 3.383 mm bevel
residence. They are upstream base material, not detached chip mass. A first
local vertical-neighbour trace does not straddle the plane and is retained.
A second trace selects actual positive-side support neighbours sharing nonzero
weight/gradient reference nodes: all 216 interfaces cross the finite edge,
152 with engaged stroke. No historical event is reconstructed from a snapshot.

Versioned opt-in **tlmpm_v7_1** shares v7 x/F/contact. It additionally completes
a digit only after a moving reference-support interface crosses the finite
edge, while that interface remains in the physical bevel, with actual
directional knife contact work at its other endpoint. It uses no task type,
target plane, selected bridge ids, wider band, particle deletion or smoothing.
This remains logical-grid cutting, not a calibrated cohesive-fracture model.

Its single primary TRIM still FAILS: 304 finite link events and 72 completed
digits do not close the material; 1 component and 0.470 mm support overlap
remain. The separate CUT regression passes with two 72 g pieces and zero
additional completed digits. No further physical alternative is invented.
Candidate v1 episode wall is 426 s TRIM / 188 s CUT; the exploratory runs use
GPU 0, so comparisons with GPU 1 are not isolated speed or numerical-repeat
claims. **The CUT profile stays bound to original v7**, never this candidate.
[Actual traces, candidate results and source hashes](../evidence/wp14/front_candidate_001.json).

After the GPU study, focused regressions reproduced pending evidence surviving
disarm, then rotation and side/band exits during final review. The lifecycle
correction is versioned as `finite_support_interface_v1_2`; it clears expired
evidence before every early return. **v1_1 and v1_2 GPU task outcomes are NOT_RUN**. The GPU
rows above remain tied to original candidate source `48c12f4b…`; no result is
reassigned to the correction. Neither experimental front enters the CUT profile.

## Physical/contact quality and unchanged surfaces

All four successful original-v7 CUT instances have finite final state,
minimum final det(F) 0.962, sampled minimum 0.940, no slot overflow, maximum
mass bookkeeping error 7.62e-13. Stability is sampled every ten controls;
this is not an all-substep bound. No particle is deleted. Final momentum,
kinetic/gravitational/projected-Hencky energy diagnostics are recorded.
Plastic/contact/grid-transfer dissipation and complete fixture work are not
closed: **total energy balance NOT_RUN**. Cumulative contact pretransfer
energy expressions are not total-system energy.

| CUT solver / dt | Control-average reaction peak (N) | Blade overlap max (mm) | Support overlap max (mm) | Tool work (mJ) |
| --- | ---: | ---: | ---: | ---: |
| v5 / 100 us | 0.04461 | 0.4833 | 0.07579 | 0.2510 |
| v5 / 50 us | 0.12150 | 0.1210 | 0.02860 | 0.7698 |
| v7 / 100 us | 0.13411 | 0.1245 | 0.02917 | 1.3694 |
| v7 / 50 us | 0.16738 | 0.0747 | 0.01856 | 1.7922 |

This is sensitivity at two dt values, **not force convergence or calibration**.
Original v7 TRIM support overlap is 0.467 mm versus baseline 0.055 mm;
CUT contact improvements do not clear that regression.
[Same-command dt pair](../evidence/wp14/sensitivity_001.json).
Six fresh CPU v7 physical negatives pass. New 1 mm GPU cut/withdraw/side-push
also passes: two components, unchanged labels/bonds through disarmed motion,
actual face distance 0.689 mm at recontact and 144 finite consistent Hermite
events. The known 2 mm CPU sever-precondition xfail is still not a pass.
[Fresh recontact](../evidence/wp14/recontact_001.json).

Both unchanged surface profiles were measured on **12 saved states / 24 meshes**.
For standard CUT, particle mesh roughness is 0.1548→0.0124 mm RMS, normal P95
26.68→2.30 degrees, locus coverage 0.953→1.000. Cut-aware roughness is
0.2522→0.0312 mm and normal P95 29.75→5.55 degrees. Actual physical-locus
whole-face plane P95 is about 1.233→0.255 mm; firm CUT 2.115→0.379 mm.
Rigid alignment is analysis only and does not flatten geometry.

Defects remain explicit: v7 standard particle mesh has one degenerate
triangle, four-slice mesh five; Partial particle coverage worsens 0.658→0.325
and has no valid core-normal/roughness samples, whereas cut-aware reaches 1.000.
Partial/TRIM have no whole-face planar penalty. Baseline TRIM cut-aware has
24 sampled intersection hits. The 512-triangle proxy misses unsampled/coplanar
defects; zero is not full-mesh validity. During frames keep their actual,
possibly different capture times. No renderer change or surface promotion follows.
[Surface measurements](../evidence/wp14/surfaces_001.json).

## Native, adoption and release gates

Four fresh Native episodes complete **4/4 TASK_SUCCESS and 4/4 HOME PASS**.
Both candidate material receipts verify exact profile configuration hash
`9a595614…` and original v7 source/front/contact. No fallback or episode swap.
Both pairs use identical canonical scene/goal/controller settings. Measured
feedback changes resolved commands: paired maximum position difference is
0.040/0.061 mm for C00/H01. This is a coupled-controller comparison, not an
identical-pose replay. Whole-episode tracking P95 is 1.37–1.40 mm, maximum
4.78–5.18 mm, including transitions; no perfect tracking claim follows.

Native position-only face plane P95 improves 1.221→0.236 mm on C00 and
2.111→0.343 mm on H01. Process wall v5/v7 is 387.11/382.70 s and
290.65/290.71 s respectively; mean material step 268/270 ms and 181/185 ms.
These include recording/coupling and do not establish a general speedup.
Native recorder has no F: Native x/F is **NOT_RUN**. Contacts, grasp, actual
camera transforms, recorded finite state, task/v1 precision and HOME remain
separate. Additional measured-joint collision replay is NOT_RUN; actual full
plan preflight passes before each launch. No Native TRIM, sensor calibration,
Isaac Lab parity or complete scientific integration acceptance is claimed.
[Native receipts](../evidence/wp14/native_001.json),
[genuine matched-time video provenance](../evidence/wp14/media_001.json).

| Family / selection | Current scope |
| --- | --- |
| General default | v5 + Law 8, unchanged |
| v7_cut_research_v1 / Straight CUT | Validated Research Profile in four measured instances; original v7 |
| Partial | Experimental; one new distinct-depth control passes |
| Multi-cut | Experimental; one four-slice pattern passes |
| TRIM / v7 and v7_1 | Unsupported by CUT profile; primary failures retained |
| Reverse/oblique | Experimental / new adoption NOT_RUN; historical slotted evidence preserved |

The explicit selection validates scene/goal/controller/discretization and pins
solver/material/workcell sources. Runtime overrides outside scope raise;
selecting a configuration is not an outcome certificate for arbitrary actions.

```python
from robotic_cutting_sim.research_profiles import compile_research_task

task = compile_research_task(
    "v7_cut_research_v1", "configs/wp12/scenes/bar_flat_board.json", "sever_mid")
# task uses the recorded path_v2 controller and a 14 s completion window.
```

Reproduce into fresh paths, with this worktree's src on PYTHONPATH:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/run_wp14_queue.py --protocol configs/wp14/core_v1.json --root artifacts/wp14/my_core
.venv/bin/python scripts/run_wp14_queue.py --protocol configs/wp14/heldout_v1.json --root artifacts/wp14/my_heldout
.venv/bin/python scripts/run_wp14_native.py --protocol configs/wp14/heldout_v1.json --root artifacts/wp14/my_native --gpu 0
.venv/bin/python -m pytest -q -rsx
```

Base versions are in each manifest: Python 3.12.12, Warp 1.17.0, Newton 1.6.1,
NumPy 2.5.4 / SciPy 1.18.1; Native Sim 6.2 uses its isolated NumPy 2.3.1 /
SciPy 1.17.0 / Torch 2.11+cu128. Two RTX 5090s, driver 570.211.01; no environment,
driver, other worktree or policy training changes. Large states and all failed
attempts stay under ignored artifacts/wp14. One initial nonpersistent shell
launch never started physics; its empty log is retained before tmux launch.

Local full CPU: **1705 passed, 9 skipped, 6 xfailed**, 1921.97 s; collected at
3ceb131 while metadata evolved, not an exact-final-head claim. After scope
restriction, **35 focused tests pass**; final current-source checks including
pending-event and effective-material regressions record **60 passed**. [Full collected receipt](../evidence/wp14/local_verification_001.json),
[latest focused checks and retained failures](../evidence/wp14/local_verification_003.json).
The sequential read-only xhigh [review](../evidence/wp14/review_001.json) found two
real issues: runtime material overrides and expired pending front events. Both
are reproduced in retained failing tests, fixed and covered by the 60 checks.
The read-only [delta review](../evidence/wp14/review_002.json) is **ACCEPT**.
A corrected-test replay against the historical candidate confirms three real
rotation/side/band failures and one already-passing outside-edge control; the
original outside-edge test error is retained and is not solver evidence.
Real media encoding and local full decode complete; five previous moving GIFs
and all seven WP-13 full MP4s remain byte-identical. Final scientific [adoption decision](../evidence/wp14/adoption_001.json) is
separate from publication. PR/main merge and exact-main CI/rendered remote
playback are release gates; their actual URLs, merge SHA and byte/playback
receipts are recorded beside the runner TASK and in the merged PR.
publication_verified and DONE may be written only after those gates pass.

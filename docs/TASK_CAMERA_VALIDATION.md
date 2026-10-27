# CUT/TRIM native camera and current-mesh audit

Both actual Isaac Sim 6.2 episodes retain 401 measured articulation states and
41 camera frames over 8 s, seed 42. This independent CPU audit reconstructs the
recorded task fixtures, lowered WORLD table, measured per-link poses, fixed
knife mount and exact current material mesh over the source USD weak layer.
CUT uses the recorded 35 mm wedge; TRIM uses the recorded 12 mm wedge. The
current-volume surface comes from current mechanical components, without future
component labels. Inputs, source snapshots and hashes are retained with each
artifact; the [public receipt](evidence/task_camera_audit.json) includes summaries,
frozen gates and exact artifact/result hashes. Audit/scorer source is committed
as `6b75f1f`.

```bash
.venv/bin/python scripts/audit_live_camera_mesh.py artifacts/isaacsim_cut01_001 artifacts/live_camera_mesh_cut01_001
.venv/bin/python scripts/audit_task_object_visibility.py artifacts/live_camera_mesh_cut01_001 artifacts/task_object_cut01_001
.venv/bin/python scripts/audit_live_camera_mesh.py artifacts/isaacsim_trim01_001 artifacts/live_camera_mesh_trim01_001
.venv/bin/python scripts/audit_task_object_visibility.py artifacts/live_camera_mesh_trim01_001 artifacts/task_object_trim01_001
.venv/bin/python -m pytest -q tests/test_camera_mesh_audit.py tests/test_task_object_visibility.py
```

The unchanged CAM-03/CAM-04 geometry gates pass: maximum frame p95 error against
actual RTX is 0.179 µm in either local camera and 0.715 µm in Global; minimum
full geometry/reference valid-mask IoU is 0.9999618 for CUT and 0.9999577 for
TRIM. Maximum material/camera/robot timestamp discrepancy is 8.9e-16 s after
accounting for the recorded 6 ms native startup origin. Rare large first-hit
mismatches remain in the receipts, including a maximum 859 mm error; the p95
gate is not an all-pixel bound. No valid tail is trimmed.

| Task | Global arm 0 min pixels | Arm 1 | Knife | Material | Table |
|---|---:|---:|---:|---:|---:|
| CUT-01 | 18,605 | 27,791 | 85 | 536 | 161,625 |
| TRIM-01 | 18,627 | 27,791 | 43 | 558 | 161,699 |

These are first-hit geometric attributions whose depths match actual RTX within
1 mm. They are evaluator diagnostics, not native semantic segmentation. Every
required Global object exceeds the frozen 25-pixel CAM-03 threshold in all
41 frames. CPU geometry audits took 57.20/55.90 s, respectively; no new renderer
or GPU job was used.

**Strict native CAM-04 remains FAIL for both tasks.** Native SingleView outputs
are kept separate from RTX reference depth. Minimum whole-frame native versus
geometry valid-mask IoUs are 0.6465/0.3369 for CUT and 0.6431/0.3089 for TRIM,
below the unchanged 0.99 gate. Native p95 is at most 0.238 µm on common valid
pixels, but full-frame hard tails reach 618/1,006 mm for CUT and 536/809 mm for
TRIM. An ideal depth replacement, validity repair, denoiser or tail filter is
never used to promote that gate. These captures use the actual version-gated
native SingleView profile with noise OFF; physical D455 calibration and noise
fidelity remain separate, unverified claims.

The task-object scorer describes the pixels where each actual current material
component and knife are first visible. Invalid native pixels are counted
separately; depth p95/max and both tail counts include every finite valid native
pixel in that ROI. RTX versus geometry mismatch counts are also retained.
ROIs are oracle evaluator data and never enter observations. These descriptive
measurements introduce no new acceptance threshold. The following values pool
all 41 frames; coverage is weighted by reference-visible pixel samples, followed
by the minimum frame coverage. Tail counts are pixel samples across frames.

| Task | Camera | Object | Coverage weighted / min | Invalid samples | Valid p95 µm | Valid max mm | >1 mm / >5 mm |
|---|---|---|---:|---:|---:|---:|---:|
| CUT-01 | d455_0 | Material | 97.53% / 89.99% | 1,021 | 0.179 | 0.004172 | 0 / 0 |
| CUT-01 | d455_0 | Knife | 100.00% / 100.00% | 0 | 0.179 | 0.000298 | 0 / 0 |
| CUT-01 | d455_1 | Material | 98.05% / 97.60% | 635 | 0.179 | 25.755882 | 70 / 1 |
| CUT-01 | d455_1 | Knife | 88.91% / 87.21% | 1,267 | 0.179 | 136.687219 | 10 / 1 |
| TRIM-01 | d455_0 | Material | 99.12% / 98.67% | 360 | 0.179 | 12.183607 | 3 / 2 |
| TRIM-01 | d455_0 | Knife | 95.18% / 90.32% | 135 | 0.298 | 63.000560 | 6 / 6 |
| TRIM-01 | d455_1 | Material | 98.18% / 97.15% | 598 | 0.179 | 5.941570 | 14 / 2 |
| TRIM-01 | d455_1 | Knife | 94.94% / 90.48% | 153 | 0.179 | 26.416898 | 12 / 12 |

CUT's two current 72 g material components are individually observed in native
depth: each reaches 517 pixels in camera 0; camera 1 reaches 339/491 pixels.
After actual separation, combined material coverage is 98.21%/97.98% weighted.
The weaker camera-0 component has a minimum frame coverage of 80.04%, which is
retained rather than hidden by the aggregate. Both all-frame and post-separation
component summaries are in the receipt. These are two substantial severed
pieces, not a claim of a small shaving chip. TRIM remains one mechanical
component and therefore has zero visible separated-component pixels; a rendered
groove does not establish detached material or successful trimming.

Four targeted CPU tests pass, including missing-pixel and large-valid-tail
retention checks. A legacy D0 regression re-audits frames 0/40 and exactly
matches the original depth and visibility metrics from
`artifacts/live_camera_mesh_ff_002`; that two-frame regression is not a new
full D0 episode. The partial CUT debug audit is retained separately as
`artifacts/live_camera_mesh_cut01_debug_001`. Camera audit commands deliberately
exit nonzero when strict native CAM-04 fails; successful execution is not a
replacement for the recorded failure.

## Seeded randomized native episodes

The actual seed42 recordings in
`artifacts/isaac_native_randomized_benchmark_001/{cut-01_randomized_seed42,trim-01_randomized_seed42}`
were audited over all41 captures each. They use a140mm bar with recorded
XY translation and3.94736° workpiece yaw; CUT samples1mm particles and TRIM
explicitly samples2mm particles, both on a4mm reference grid. TRIM contains
three continuous bumps, including an irregular middle bump. The audit reads
these recorded frames, fixtures, blade dimensions and meshes instead of
regenerating geometry from current task defaults.

The exact inputs, frozen gates, per-current-component visibility and media
hashes are in
[`task_randomized_camera_audit.json`](evidence/task_randomized_camera_audit.json).
CPU source snapshots and full records remain in
`artifacts/live_camera_mesh_{cut,trim}01_randomized_seed42_001` and
`artifacts/task_object_visibility_{cut,trim}01_randomized_seed42_001`.
No simulator or native renderer was rerun for these audits.

| Randomized case | Geometry bridge | Time synchronization | Global CAM03 | Existing separated-component visibility | Strict native CAM04 |
|---|---|---|---|---|---|
| CUT-01 |PASS|PASS|PASS|PASS:416 native pixels|**FAIL**|
| TRIM-01 |PASS|PASS|PASS|**FAIL**:10 pixels for the smallest current component|**FAIL**|

The largest per-frame geometry/RTX p95 differences remain0.179µm in either
local camera and0.715µm in Global; minimum geometry/RTX valid-mask IoU is
0.9999663. Maximum material/camera/robot timestamp span is8.89e-16s. These
are p95 and validity gates, not all-pixel error bounds; all large first-hit
and native-depth tails remain recorded.

| Randomized task | Global arm0 min pixels | Arm1 | Knife | Material | Table |
|---|---:|---:|---:|---:|---:|
| CUT-01 |18,478|27,791|81|483|161,857|
| TRIM-01 |18,426|27,791|44|475|161,989|

Whole-image native valid-mask IoUs remain far below the unchanged0.99 gate:

| Case | Camera | Native mask IoU range | All-material native coverage | Material valid maximum error |
|---|---|---:|---:|---:|
| CUT-01 |D4550|.6486–.6581|96.5295%|6.1323mm|
| CUT-01 |D4551|.3371–.3404|98.3853%|2.5811mm|
| TRIM-01 |D4550|.6458–.6550|98.0006%|25.9428mm|
| TRIM-01 |D4551|.3116–.3402|98.0308%|21.5078mm|

The coverage values pool all41 frames within measured-mesh material ROIs.
Whole-frame valid native error tails reach631/783mm for CUT and617/1,057mm
for TRIM. Native noise is OFF; holes and every finite valid tail are retained,
with no ideal replacement or observation repair. Local material coverage does
not establish complete image quality or calibrated D455 fidelity.

CUT produces two actual substantial pieces of66.6g and59.4g. In the14 captures
with the final two-component state (5.4–8s), their native-valid pixel ranges
are428–475 /220–409 in D4550 and235–338 /412–416 in D4551. These are severed
parts, rather than small shaving chips.

TRIM produces three actual chip components plus a126.096g substrate. Its
task-flatness metrics remain FAIL. The final four-component phase spans17
captures (4.8–8s); each chip is geometrically visible in every capture:

| Actual chip mass | Particles | D4550 valid pixels | D4551 valid pixels | Global ideal pixels |
|---|---:|---:|---:|---:|
| .672g |56|9|9|5|
| .768g |64|12|12|8|
|1.044g |87|11–15|12–17|7–11|

The two smaller chips have100% native ROI coverage in both local cameras in
this final phase. The87-particle chip has98.73%/97.90% coverage, retaining one
21.508mm valid D4551 outlier. Across every actual current detached chip in all
41 frames, the largest native-visible single chip reaches15 pixels in D4550
and17 in D4551. Thus visibility remains below25 pixels even when all chips
are evaluated, rather than only the existing gate's smallest component. Chip
pixels are not summed to redefine that criterion. Component IDs are interpreted
with each frame's current particle count; early component1 can still be the
connected substrate and is not labelled as a future detached chip.

Each case also has six captured-array videos and its task-plane trajectory in
its raw `media_001` directory: Global, two RGB, two measured-depth and
synchronized multiview. All12 videos were verified as41 frames at5fps, with
source hashes in the receipt. No additional media was copied to public proof
directories. Formal observations retain native depth and validity; all oracle
ROI and component measurements remain evaluator-only.

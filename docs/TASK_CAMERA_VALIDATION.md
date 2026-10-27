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

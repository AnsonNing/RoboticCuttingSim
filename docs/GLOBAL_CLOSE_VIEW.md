# Close target view from the fixed Global D455

The Global D455 now looks directly at the material from `(0.28,-0.42,0.46)` m,
aimed at `(0,0,0.02)` m: approximately 0.670 m from the target. This is a close
cutting view. Parts of the table and robots are intentionally outside the image.
The previous full-workcell views and their recordings remain preserved.

Two new **actual Isaac Sim 6.2** episodes each completed 400 control intervals,
4,000 native rigid steps and 8 simulated seconds, with 41 synchronized RGB-D
captures and 401 measured hand/camera samples. Independent recorded-state audits
verify their camera geometry and recompute their physical Task scores. Their
[compact receipt](evidence/global_close_view.json) retains source/input hashes,
unchanged criteria and separate failures.

## Actual improvement

The table reports the minimum across all 41 recorded frames. Material and knife
counts use actual valid native depth agreeing within the unchanged 2 mm tolerance
with full current-scene first hits from measured robot poses and the current
material mesh. This is geometric attribution checked against native depth,
not a native semantic segmentation output.

| Native recording | Global valid depth | Material pixels | Knife pixels | Material gain over v2 |
| --- | --- | --- | --- | --- |
| Preserved v2 CUT | 15.8337% | 108 | 19 | Reference |
| Close v3 CUT | 27.0527% | 878 | 102 | 8.13× |
| Preserved v2 TRIM | 15.8490% | 112 | 9 | Reference |
| Close v3 TRIM | 27.2327% | 999 | 71 | 8.92× |

All Global visibility, timestamps and measured-pose checks pass under the
existing engineering protocol. Each current separated material component has
depth-consistent pixels in at least one camera at every recorded post-split
frame. The complete TRIM camera screen still fails because its secondary wrist
loses sight of the withdrawn knife, as in the preserved v2 recording.

| Actual close-view recording | Global RGB | Global depth | Three synchronized views |
| --- | --- | --- | --- |
| CUT-01 | [Video](media/global_close_cut01/global_d455.mp4) | [Video](media/global_close_cut01/global_d455_depth.mp4) | [Video](media/global_close_cut01/synchronized_multiview.mp4) |
| TRIM-01 | [Video](media/global_close_trim01/global_d455.mp4) | [Video](media/global_close_trim01/global_d455_depth.mp4) | [Video](media/global_close_trim01/synchronized_multiview.mp4) |

## Geometry and sensor scope

[Rig v3](../configs/wrist_camera_rig_v3.json) changes the fixed Global placement
and viewing role. The official D455 asset, native 640×480 resolution, actual
runtime intrinsics (`fx=fy≈317.0431`, `cx=320`, `cy=240`) and public optical-depth
range 0.6–6 m are retained. No virtual optical zoom or sensor ROI is substituted.
Render clipping remains 0.001–20 m, so near occluders and RGB outside the useful
depth range remain visible before depth-validity filtering.

Both D405 cameras remain actual hand children with fixed optical mounts and
848×480 ideal prototype streams. Their transforms, optics and depth ranges are
unchanged. Global remains World-fixed. The native episodes retain the explicit
noise-off SingleView recipe; neither that pipeline nor the D405 prototypes are
physically calibrated hardware models. See [sensor boundaries](WRIST_CAMERAS.md).

The full 401-pose visual-surface audit finds zero Global-housing/robot or
wrist-housing/robot intersections. The primary visual mounting rod still
intersects its own link7 flange: 74 triangles at each sampled pose. The unchanged
strict mount screen therefore remains **FAIL**. These are sampled surface checks;
camera contact physics, closed-volume containment and continuous swept
collision are not validated.

Native depth holes and outliers are retained. The worst whole-frame valid-mask
IoU against current geometry is 0.96697 for CUT and 0.96571 for TRIM, below the
original CAM-04 0.99 mask screen. Maximum retained depth discrepancies reach
212.7 mm and 608.1 mm respectively. Engineering visibility does not establish
complete depth-accuracy or hardware-model acceptance.

## Physical regression and reproduction

The independent unchanged Task evaluator recomputes CUT as **PASS_TASK_METRICS**:
one continuous 144 g bar becomes two 72 g components. TRIM again detaches a
1.608 g chip and removes 98.5294% of the reference above-plane material, while
its strict surface score remains **FAIL_TASK_METRICS**. Both retain particle
count, reference mass and permanent cut history, with separation recorded for
2.4 seconds after withdrawal. Partitioned coupling-work residuals and incomplete
native contact/joint-limit audit coverage remain separate limitations.

The two Lab close-view checks each run only two control steps and three camera
captures. Their measured-pose/optics checks pass; Task success is **NOT_RUN**.
They are not new full Lab cutting episodes. All prior Lab and fixed-camera
evidence remains its original version.

Use the installed runtime and fresh output directories:

```bash
.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id CUT-01 --steps 400 --gpu 1 --device cuda:1 \
  --camera-layout wrist_d405_global_d455 --camera-mode native_single_view \
  --native-noise off --output-dir artifacts/global_close_cut_NEW

.deps/isaacsim-venv/bin/python scripts/run_isaac_sim_cutting.py \
  --task-id TRIM-01 --particle-spacing 0.002 --steps 400 --gpu 1 --device cuda:1 \
  --camera-layout wrist_d405_global_d455 --camera-mode native_single_view \
  --native-noise off --output-dir artifacts/global_close_trim_NEW
```

Raw full native recordings are `artifacts/isaacsim_global_close_cut01_001` and
`artifacts/isaacsim_global_close_trim01_001`. Independent camera, Task and
sampled-clearance receipts are in `artifacts/global_close_camera_audit_001`,
`artifacts/global_close_task_audit_001` and
`artifacts/global_close_clearance_audit_001`. No solver, policy or camera was
advanced by these offline audits; the complete benchmark remains unfinished.

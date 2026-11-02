# Three Global views and real wrist RGB-D gallery

Every current movie shows **LEFT rig10 / RIGHT rig11 / OPPOSITE TOOL rig14 RGB**, then **arm_0 / arm_1 Wrist D405 RGB and Depth**. The three fixed external cameras use 0.60 m distance, 24° horizontal FOV and 17.12394° elevation. Opposite-tool has zero horizontal obliquity. RGB foreground clipping is 0.001 m; actual robots and occluders stay visible.

The Global panels are exact saved-mesh / measured-pose Native RTX **camera replays**, with zero physics steps. Wrist RGB-D is the original capture at the same physical timestamp. Wrist depth is optical Z in meters, 0.07–0.50 m; blue→green→red, invalid black. Narrow Global stereo remains **BLOCKED** and is not published as calibrated depth. Camera models remain simulated/provisional.

There are **47 full MP4s and eight moving README GIFs**: 44 named episode slots plus three comparisons, backed by **37 unique state streams and 111 camera-only replays**. Nine streams reuse historical exact states, 17 reuse prior camera-task Native recordings, nine needed one minimal controlled-optics recording each, and two new current registered demonstrations serve four unavailable historical-source slots. Seven slots share an identical source; these are not extra trials. All new recordings are engineering media work, outside frozen scientific denominators. Historical sources retain their actual calibration; current recordings retain v5 fixed wrist mounts and arm_1 safe park.

[Camera directions, calibration and runnable record/display API](GLOBAL_CAMERA_VIEWS.md) · [All synchronization, encoder/source hashes and timing receipts](evidence/three_global/media_001.json) · [37-source inventory and readback](evidence/three_global/source_inventory_001.json) · [Controls, failures and limitations](reports/THREE_GLOBAL_CAMERA_REFRESH.md).

[Complete prior gallery and original outcomes, pinned before this refresh](https://github.com/AnsonNing/RoboticCuttingSim/blob/b9100bcd662bafcb81cefedae424856d047e9e72/docs/NATIVE_MULTIVIEW_GALLERY.md). All prior media/evidence files remain unchanged. No current replay uses the prior Hero's crop or presentation close-up.

## Sequential two-protrusion demo

| Episode | Source kind | Actual source task / HOME | Full seven-pane MP4 |
| --- | --- | --- | --- |
| two_bumps_v7_2_native | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/two_bumps_v7_2_native.mp4) |
| two_bumps_original_v7_failure | Prior camera-task engineering recording | FAIL / PASS | [MP4](media/three_global_v1/two_bumps_original_v7_failure.mp4) |

Versioned v7_2: A/B 100% detached, bodies 1 → 2 → 3, base intact, HOME PASS. The original v7 failure remains visible: A stays connected, B detaches. Shavings can rest on the substrate. [Actual state arrays](evidence/three_global/native_two_bumps_states_v1.npz) · [Independent mechanical graph and mass proof](evidence/three_global/native_two_bumps_detachment_001.json) · [Eight actual cutting/withdrawal/post-cut frames](evidence/three_global/visibility/contact_sheet.png). This audits one source, not a new success-rate denominator.

## Current CUT / TRIM / multi-cut / scene

| Episode | Source kind | Actual source task / HOME | Full seven-pane MP4 |
| --- | --- | --- | --- |
| wp13_v5_cut | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp13_v5_cut.mp4) |
| wp13_v7_cut | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp13_v7_cut.mp4) |
| wp13_v5_trim | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp13_v5_trim.mp4) |
| wp13_v7_trim | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp13_v7_trim.mp4) |
| wp13_v5_multicut | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp13_v5_multicut.mp4) |
| wp13_v7_multicut | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp13_v7_multicut.mp4) |
| wp14_v5_standard_cut | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp14_v5_standard_cut.mp4) |
| wp14_v7_standard_cut | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp14_v7_standard_cut.mp4) |
| wp14_v5_firm_offset_cut | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp14_v5_firm_offset_cut.mp4) |
| wp14_v7_firm_offset_cut | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp14_v7_firm_offset_cut.mp4) |
| wp14_walkthrough | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/wp14_walkthrough.mp4) |

## Historical gallery slots

| Episode | Source kind | Actual source task / HOME | Full seven-pane MP4 |
| --- | --- | --- | --- |
| archive_demos_demo1a_01_law8_final_demo1a | Prior camera-task engineering recording | PASS_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_demos_demo1a_01_law8_final_demo1a.mp4) |
| archive_demos_demo1a_02_pathv2_d_demo1a | Prior camera-task engineering recording | PASS_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_demos_demo1a_02_pathv2_d_demo1a.mp4) |
| archive_demos_demo1b_03_law8_final_demo1b | NEW minimal engineering recording | FAIL_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_demos_demo1b_03_law8_final_demo1b.mp4) |
| archive_demos_demo1b_04_pathv2_d_demo1b | NEW minimal engineering recording | FAIL_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_demos_demo1b_04_pathv2_d_demo1b.mp4) |
| archive_demos_demo2_05_law8_final_demo2 | NEW minimal engineering recording | FAIL_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_demos_demo2_05_law8_final_demo2.mp4) |
| archive_demos_demo2_06_pathv2_c_demo2 | NEW minimal engineering recording | FAIL_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_demos_demo2_06_pathv2_c_demo2.mp4) |
| archive_demos_demo3_07_law8_final_demo3 | NEW minimal engineering recording | PASS_SLICE_PUSH / NOT_REPORTED | [MP4](media/three_global_v1/archive_demos_demo3_07_law8_final_demo3.mp4) |
| archive_demos_demo3_08_vision_e_demo3 | NEW minimal engineering recording | PASS_SLICE_PUSH / NOT_REPORTED | [MP4](media/three_global_v1/archive_demos_demo3_08_vision_e_demo3.mp4) |
| archive_deployment_cut01_09 | **NEW current representative; historical raw unavailable** | PASS_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_deployment_cut01_09.mp4) |
| archive_deployment_trim01_10 | **NEW current representative; historical raw unavailable** | FAIL_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_deployment_trim01_10.mp4) |
| archive_global_close_cut01_11 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_global_close_cut01_11.mp4) |
| archive_global_close_trim01_12 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_global_close_trim01_12.mp4) |
| archive_wp08_physical_grasp_v2_cut01_13 | **NEW current representative; historical raw unavailable** | PASS_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_wp08_physical_grasp_v2_cut01_13.mp4) |
| archive_wp08_physical_grasp_v2_trim01_14 | **NEW current representative; historical raw unavailable** | FAIL_TASK_METRICS / NOT_REPORTED | [MP4](media/three_global_v1/archive_wp08_physical_grasp_v2_trim01_14.mp4) |
| archive_wp11_slice_push_relocate_15 | NEW minimal engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp11_slice_push_relocate_15.mp4) |
| archive_wp11_test_a_s6_16 | NEW minimal engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp11_test_a_s6_16.mp4) |
| archive_wp11_test_b_s1502_17 | NEW minimal engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp11_test_b_s1502_17.mp4) |
| archive_wp11_trim_rho10_18 | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp11_trim_rho10_18.mp4) |
| archive_wp11_trim_rho10_clear_19 | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp11_trim_rho10_clear_19.mp4) |
| archive_wp12_demos_demo1_clean_cut_particle_surface_v1_20_demo | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp12_demos_demo1_clean_cut_particle_surface_v1_20_demo.mp4) |
| archive_wp12_demos_demo3_trim_rho10_particle_surface_v1_21_demo | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp12_demos_demo3_trim_rho10_particle_surface_v1_21_demo.mp4) |
| archive_wp12_demos_demo4_four_slices_particle_surface_v1_22_demo | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp12_demos_demo4_four_slices_particle_surface_v1_22_demo.mp4) |
| archive_wp12_surface_live_cut_23 | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp12_surface_live_cut_23.mp4) |
| archive_wp12_surface_live_test_b_24 | Prior camera-task engineering recording | PASS / PASS | [MP4](media/three_global_v1/archive_wp12_surface_live_test_b_24.mp4) |
| archive_wrist_cut01_25 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_wrist_cut01_25.mp4) |
| archive_wrist_motion_probe_26 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_wrist_motion_probe_26.mp4) |
| archive_wrist_trim01_27 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_wrist_trim01_27.mp4) |
| archive_wrist_v4_cut01_28 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_wrist_v4_cut01_28.mp4) |
| archive_wrist_v4_trim01_29 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_wrist_v4_trim01_29.mp4) |
| archive_wrist_v5_cut01_30 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_wrist_v5_cut01_30.mp4) |
| archive_wrist_v5_trim01_31 | Historical exact-state replay | NOT_REPORTED / NOT_REPORTED | [MP4](media/three_global_v1/archive_wrist_v5_trim01_31.mp4) |

The four deployment/wp08 slots marked NEW representative cannot restore the missing original raw geometry/robot state. Their filenames retain inventory correspondence only: they show two clearly documented new current CUT-01/TRIM-01 sources, not those historical controllers, trials or outcomes. Their original movies and failures remain in the pinned gallery. Nine other new recordings reuse the prior specified input recipes but have new physical trajectories; only the three camera replays share exactly the same trajectory. Historical scientific results must be read in the original reports, not replaced with the new table's outcomes. Missing HOME metadata means NOT_REPORTED, not PASS.

The earlier WP-11 sources had 6/6/28 missing Wrist0 captures and no exact per-frame mesh; their old failure/gap receipts remain. The three necessary new controlled-optics recordings retain full original-clock wrist streams. The three viewpoints do not triple the material experiments.

## Matched captured-time comparisons

| v5 left / v7 right, each with all seven panes | Full MP4 |
| --- | --- |
| wp13_cut_compare | [MP4](media/three_global_v1/wp13_cut_compare.mp4) |
| wp14_standard_cut_compare | [MP4](media/three_global_v1/wp14_standard_cut_compare.mp4) |
| wp14_firm_offset_cut_compare | [MP4](media/three_global_v1/wp14_firm_offset_cut_compare.mp4) |

All input capture timestamps and every encoded VFR PTS match. These are separately coupled Native runs, not identical measured-feedback commands. Full movies play at physical 1× time. README GIFs use the stated preview speed:1× except the labelled four-slice3× preview, at a smaller presentation size. No comparison changes frozen scoring.

## Archives without the required raw cameras

These earlier recordings used a different fixed-camera layout. Their original files remain; no fixed camera is relabelled as a wrist. Re-recording them would be a new episode rather than restoration of historical evidence.

| Original file | Available raw streams / reason |
| --- | --- |
| `docs/media/cut01/d455_0_depth.mp4` | d455_1, d455_0 |
| `docs/media/cut01/d455_0_rgb.mp4` | d455_1, d455_0 |
| `docs/media/cut01/d455_1_depth.mp4` | d455_1, d455_0 |
| `docs/media/cut01/d455_1_rgb.mp4` | d455_1, d455_0 |
| `docs/media/cut01/global.mp4` | d455_1, d455_0 |
| `docs/media/cut01/synchronized_multiview.mp4` | d455_1, d455_0 |
| `docs/media/live_isaac_kit_d0/d455_0.mp4` | d455_1, d455_0 |
| `docs/media/live_isaac_kit_d0/d455_1.mp4` | d455_1, d455_0 |
| `docs/media/live_isaac_kit_d0/global.mp4` | d455_1, d455_0 |
| `docs/media/live_isaac_kit_d0/measured_depth.mp4` | d455_1, d455_0 |
| `docs/media/live_isaac_kit_d0/synchronized_multiview.mp4` | d455_1, d455_0 |
| `docs/media/trim01/d455_0_depth.mp4` | d455_1, d455_0 |
| `docs/media/trim01/d455_0_rgb.mp4` | d455_1, d455_0 |
| `docs/media/trim01/d455_1_depth.mp4` | d455_1, d455_0 |
| `docs/media/trim01/d455_1_rgb.mp4` | d455_1, d455_0 |
| `docs/media/trim01/global.mp4` | d455_1, d455_0 |
| `docs/media/trim01/synchronized_multiview.mp4` | d455_1, d455_0 |

The four kinematic diagnostics have no Native wrist/global recording and remain labelled kinematic. They are not counted as missing or successful Native demonstrations.

## Reuse one exact recording without another material run

```bash
.venv/bin/python scripts/render_native_global_views.py /path/to/episode_000 \
  --gpu 0 --out artifacts/three_view_display_NEW
```

Record with supported Global optics and `--save-render-surfaces` only when exact saved state is absent. The wrapper retains actual wrists and invokes the existing renderer/encoder. Current direct narrow Native RGB-D remains guarded. Every required capture index, physics/surface/extrinsics clock, exact staged mesh hash, measured robot matrix, fixed Global K/T and hand-local wrist composition is checked. Missing exact meshes are refused; no position-only substitute, smoothing or particle deletion is used.

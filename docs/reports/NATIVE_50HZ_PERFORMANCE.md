# Native 50 Hz performance

The matched runs use Isaac Sim 6.2.0.0 and Warp 1.17.0 on the same RTX 5090
(physical GPU 1, driver 570.211.01). Each control interval is 20 ms with ten
PhysX substeps. CUT runs the same measured-edge waypoint controller at seed
42 for 14.0 simulated seconds plus 0.8 seconds of HOME. TRIM runs the frozen C3
controller on scene seed 512 and simulator seed 42 for 24.0 seconds. Its
published controller hash is
`10e128c57627a6f304471c7d9c9ddeecf7b2206a10add218776ce039f8efa506`.

Reference uses 1 mm particles on a 4 mm grid with a 0.10 ms material step. Turbo
uses 2 mm particles on the same grid with a 0.20 ms material step. Both record
the initial image and each subsequent control step. CUT keeps the two wrist
RGB-D cameras, Global RGB-D and two RGB presentation views. TRIM keeps rig14
Global RGB and Wrist-0 RGB-D.

| Backend / task / fidelity | Envs | Camera Hz | Record Hz | Cold process wall (s) | Warm episode wall (s) | Episodes/GPU-hour | Dataset integrity | Physical outcome |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Native CUT Reference, baseline | 1 | 50 | 50 | 876.23 | — | 4.11 | PASS, 741 per stream | PASS, HOME PASS |
| Native CUT Turbo, baseline | 1 | 50 | 50 | 656.12 | — | 5.49 | PASS, 741 per stream | PASS, HOME PASS |
| Native TRIM Reference, baseline | 1 | 50 | 50 | 1,004.95 | — | 3.58 | PASS, 1,201 per stream | PASS, HOME NOT_RUN |
| Native TRIM Turbo, baseline | 1 | 50 | 50 | 789.76 | — | 4.56 | PASS, 1,201 per stream | PASS, HOME NOT_RUN |
| Native CUT Reference, tuned | 1 | 50 | 50 | 334.53 | — | 10.76 | PASS, 741 per stream | PASS, HOME PASS |
| Native CUT Turbo, tuned | 1 | 50 | 50 | 227.01 | — | 15.86 | PASS, 741 per stream | PASS, HOME PASS |
| Native TRIM Reference, tuned | 1 | 50 | 50 | 534.26 | — | 6.74 | PASS, 1,201 per stream | PASS, HOME NOT_RUN |
| Native TRIM Turbo, tuned | 1 | 50 | 50 | 306.04 | — | 11.76 | PASS, 1,201 per stream | PASS, HOME NOT_RUN |
| Native CUT Turbo, collect five, recycle every four | 1 | 50 | 50 | 263.23 to first stored episode | 191.01 median | 16.74 | PASS, 5 × 741 per stream | 5 × PASS, HOME PASS |

The baseline processes come from `da120b39997d3b5bdda1492dcd8fbff0e6ead1b8`.
Cold process wall includes preflight and setup, reset, the full controller,
raw recording finalize and shutdown. The separate consolidated dataset
export took 57.21 s for CUT Reference, 57.24 s for CUT Turbo, 84.22 s
for TRIM Reference and 82.64 s for TRIM Turbo. The two CUT cameras that
only supply presentation views are included in the raw recording.

### Baseline evidence

- All recorded sensor rows passed the 50 Hz audit with zero invalid or
  mismatched timestamps. The audit checks the actual files and aligns
  frames with measured robot and tool state. CUT’s original presentation
  archives include the task clock. The calibrated camera archives also
  include the synchronized Native PhysX clock.
- Rendering took 249.08 s for CUT Reference, 245.20 s for CUT Turbo and
  197.71 s / 195.57 s for TRIM Reference / Turbo. TRIM Reference did 1,202 capture calls for
  1,201 archived timestamps. CUT Reference spent another 138.25 s in raw
  recording finalize, including the legacy stacked physics archive.
- The old TRIM media encoder produced 602 decoded pictures from 1,201
  original captures. That failed attempt and the complete source recording
  are preserved. Re-encoding the saved frames took 40.17 s for Reference and
  37.98 s for Turbo and retained all 1,201 pictures with matching timestamps.
- The grasp traces from both CUT baselines contain 7,400 valid, fresh
  coupled PhysX samples in order, plus 1,000 initial settling records.
  The existing 5 Hz readback-only attempt remains negative evidence.
  [High-speed research history](HIGH_SPEED_RESULTS.md)

Commands, runtime versions, the GPU and RSS samples, profiler stages,
audits and original recordings are under
`/home/ningan/RoboticCuttingSim-high-speed/.native50hz/artifacts/p0_50hz_001`.
Each `*_process.json` records its complete launch command. The
`*_cadence_v2.json` receipts enumerate every acquired camera stream.
`matched_phase_summary.jsonl` includes the StageProfiler and I/O
breakdowns and `EVIDENCE_SHA256SUMS` records the evidence digests.

### Measured pipeline changes

- At 13 matched CUT physics states, one and five RTX passes had exactly
  matching calibrated camera poses, intrinsics, frame timestamps and
  depth validity masks. The maximum frame-wise mean RGB difference was
  0.448/255. The minimum PSNR was 46.9 dB and the 99th-percentile absolute
  depth difference was zero at each state and camera. The probe verifies
  that later renders leave an earlier owned snapshot intact. Each episode
  retains five initial render passes.
- The writer appends camera index rows, checkpoints the written prefix
  and publishes the legacy manifest after the bounded queue drains. A
  60-file CUT codec trial measured 1.07 s and 13.60 MB for default
  DEFLATE, 0.31 s and 16.88 MB for level 1, 1.16 s and 13.47 MB for
  chunked NPZ, and 1.22 s for a stored intermediate followed by conversion.
  Level 1 preserves the arrays exactly.
- Tuned CUT Turbo shortened cold process wall by 65.4%. RTX render time
  fell from 245.20 to 40.25 s, with 745 render passes for 741 timestamps.
  The measured CUDA-to-host conversion took 3.35 s. The synchronous
  transfer still produces each snapshot at its matching physics time.
  The camera writer peaked at 23.6 MiB of the 64 MiB budget and waited
  0.0043 s for room. Streamed physical samples reduced finalize time
  from 20.23 s to 0.96 s. The separate dataset export took 28.20 s.
- Tuned CUT Reference shortened cold process wall by 61.8% with the
  original solve. TRIM Reference shortened it by 46.8% and TRIM Turbo
  by 61.2%. Each tuned TRIM episode captured 1,201 timestamps with
  1,205 render passes. Both H.264 files retained all 1,201 decoded
  frames at their original times.
- The eight-iteration CUT IK candidate preserved all 700 baseline
  target joint solutions in an offline pose replay. The live CUT Turbo
  run retained 7,400 fresh substep samples with bilateral loaded handle
  contact and passed HOME. Live IK time fell from 10.14 to 2.35 s for
  Turbo and from 10.16 to 2.51 s for Reference. The physics cadence
  stayed at ten substeps. Isolated contact and robot readback times
  varied across the matched runs.

The tuned runs and component measurements are in sibling artifact folders
`tuned_50hz_001`, `p1_render_probe_001` and `p2_codec_probe_002`.

### Resident collection

The five CUT episodes used two processes and took 1,075.16 s including
validation and consolidated export. Warm wall measures reset through export.
The first reset took 44.47 s. The three resident resets took 6.62–7.08 s.
The second process started from a warm disk cache and reset in 9.20 s.
The sustained rate was 3.32× the 50 Hz Turbo baseline plus its separate
dataset export. Compared with the tuned cold benchmark plus export, the
sustained rate was 1.19×. Host RSS peaked at 15.93 GiB including export
and validation. Sampled GPU memory peaked at 7,331 MiB.

The `rcs_dataset_v1` dataset is in `p4_full_collection_001/collect-r4full-42`
under the artifact root. The existing reader passed all five episodes.
Each episode retains five original camera streams. The consolidated
`episode_export.npz` retains the three policy camera streams with
calibration and the complete control and HOME timeline. The optional
presentation views also have static USD calibration in the runtime setup
receipts. The historical `POLICY_READY` score failed the reset tool-rest
check and requires a failed episode in the collection. Dataset integrity
and physical outcomes passed independently.

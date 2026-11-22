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

The baseline processes come from `da120b39997d3b5bdda1492dcd8fbff0e6ead1b8`.
Cold process wall includes preflight and setup, reset, the full controller,
raw recording finalize and shutdown. The separate consolidated dataset
export took 57.21 s for CUT Reference, 57.24 s for CUT Turbo and 84.22 s
for TRIM Reference. The two CUT cameras that only supply presentation
views are included in the raw recording.

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
  are preserved. A 40.17 s re-encode of the original TRIM Reference frames
  produced all 1,201 pictures with matching decoded timestamps.
- The grasp traces from both CUT baselines contain 7,400 valid, fresh
  coupled PhysX samples in order, plus 1,000 initial settling records.
  The existing 5 Hz readback-only attempt remains negative evidence.
  [High-speed research history](HIGH_SPEED_RESULTS.md)

Commands, runtime versions, the GPU and RSS samples, profiler stages,
audits and original recordings are under
`/home/ningan/RoboticCuttingSim-high-speed/.native50hz/artifacts/p0_50hz_001`.
Each `*_process.json` records its complete launch command. The
`*_cadence_v2.json` receipts enumerate every acquired camera stream.

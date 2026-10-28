# WP-10B — live policy-interface validation

Status: IN PROGRESS. Results are recorded separately from task success.

The server has two RTX 5090s and the pinned Native/Lab environments. New runs
use GPU 1; no existing process or driver is changed. The initial full CPU suite
at `1fedbadd3798572769f57dd8c4f85215f686e33d` passes 454 tests, skips 2, and
takes 89.39 s (`artifacts/wp10/baseline_cpu_001`). Gymnasium was absent from
the CPU and Native environments; `gymnasium==1.4.0` is now an explicit policy
extra. The quick suite including Gym passes 441 tests (18 CPU simulations
deselected) in 5.57 s after the interface fixes.

`scripts/run_wp10_live.py --preset policy` selects physical grasp, pregrasped
initialization, 0.0112 m initial fingers, bounded deltas, rejection/hold and,
for Lab, finite pad contacts and a 0.01 rad orientation gate. Legacy defaults
are unchanged. It records submitted/resolved/accepted commands, measured hand
and free-tool poses, IK status, raw synchronized RGB-D when requested, material
history, frozen oracle and Research evaluation in a fresh directory.

Initial Native state/Gym smoke executes two real steps with zero IK rejections;
the unavailable cut digit is rejected without consuming simulation time
(`artifacts/wp10/b_native_api_smoke_002`). Its maximum target-to-tool position
error is 0.37335 mm; no task completion is claimed for a two-step hold.
Lab smoke 001 finds an actual interface bug: Lab emits a string grasp status
while Native emits a mapping. The wrapper now accepts both formats, with
regression tests; the failed trace remains at
`artifacts/wp10/b_lab_api_smoke_001`. Native harness startup 001 also remains
preserved (Path serialization error before any physics).

Commands, source snapshots, commit, seed and elapsed time are in each
`artifacts/wp10/*/command.json`; experiment `result.json` is authoritative,
because Kit can exit with code zero during shutdown even after an exception.
The larger declared fuzz, Gym and legacy regression matrix is pending.

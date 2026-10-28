# WP-10B — live policy-interface validation

**PASS for the isolated bounded-action interface, legacy regression and Gym;
FAIL for sustained physical holding in the Lab rotation stress pilot.**
These are separate results. Exact per-run commits, commands, source snapshots,
seeds, wall times and hashes are in
[the compact receipt](../evidence/wp10/wp10b_live_validation_v1.json).

| Check (seeds 42/43/44 where repeated) | Native PhysX | Isaac Lab / Newton | Result |
| --- | --- | --- | --- |
| Identical bounded fuzz, explicit fixed tool / 4 mm | 1,200 steps / 126.63 s | 1,200 steps / 282.73 s | PASS: 1,200 identical requests and IK statuses, all <=2 mm / 2 deg |
| Separate impossible-target IK probe | 2/2 rejected / 15.34 s | 2/2 rejected / 20.26 s | PASS; absolute clamping disabled only for this isolated IK-layer probe |
| Legacy reference CUT, defaults / action=None | 400 steps / 272.06 s, strict PASS | 400 / 387.39 s, strict PASS | PASS regression |
| Legacy reference TRIM, defaults / action=None | 400 / 354.61 s, strict FAIL | 400 / 354.51 s, strict FAIL | PASS reproduction of the original 1 mm task failure |
| Unavailable cut digit | Rejected before time advances | Rejected before time advances | PASS |
| Physical-grasp Gym / state | 400 / 269.38 s | 400 / 275.64 s | PASS interface; hold is not task success |
| Physical-grasp Gym / vision_proprio | 400 / 413.49 s | 400 / 463.51 s | PASS interface, actual three-camera RGB-D |
| Separate physical-grasp balanced stress | 1,200 / 848.73 s | 209 total steps / 183.34 s, loss at 99/102/8 | Native holding PASS; Lab holding FAIL, normal loss termination |

The fixed-tool fuzz isolates action parsing and IK; it is not robust arbitrary
grasp evidence. Its cutting switch stays off. The physical stress failure is
preserved in full and cannot be counted as a task pass. An earlier continuous
rotation pilot has 12 Lab IK rejections in seed 43 before loss. Initial aborted
surface-rebuilding and continuous-rotation pilots have explicit abort receipts
and do not count toward the successful 1,200-step comparison.

After installing the missing Gym extra and adding the live/ruler regressions,
the complete CPU suite is **461 PASS / 0 SKIP in 92.45 s**
(`artifacts/wp10/cpu_full_002`). V1 `tlmpm.py`, task protocol and strict
evaluator bytes equal handoff commit `1fedbadd3798572769f57dd8c4f85215f686e33d`;
their SHA-256 values are in the receipt. The earlier baseline follows below.

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
The declared fuzz, Gym and legacy matrix is complete. Reproduce with a fresh
output path; replace the interpreter/runtime for Lab:

```bash
.deps/isaacsim-venv/bin/python scripts/run_wp10_live.py \
  --runtime isaac_sim --preset policy --controller fuzz --fuzz-pattern balanced \
  --tool-attachment-mode legacy_fixed_tool --spacing .004 --episodes 3 \
  --gpu 1 --out artifacts/wp10/reproduce_fuzz_NEW
.deps/isaacsim-venv/bin/python scripts/run_wp10_live.py \
  --runtime isaac_sim --preset policy --controller gym_hold --gym \
  --observation vision_proprio --gpu 1 --out artifacts/wp10/reproduce_gym_NEW
```

Kinematic rerun 003 found missing Gymnasium; rerun 004 completes all applicable
controllers. The shortened TRIM pass now preserves separation; strict coarse
surface accuracy remains FAIL. Supplemental
[TRIM](../media/wp10_kinematic/trim01r_skim_002.mp4),
[CUT-02 agent API](../media/wp10_kinematic/cut02_agent_002.mp4) and
[oblique CUT](../media/wp10_kinematic/cut01r_oblique20_002.mp4) movies contain no
robot and do not substitute actual live controller evidence.

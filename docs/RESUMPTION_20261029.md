# Interrupted-work resumption — 2026-10-29 Asia/Seoul

The checkout resumed at `2b45a50` on `wp08/physical-tool-grasp`, with four
unfinished tracked documentation/evidence changes and several completed actual
experiments that were still described as running. Latest remote main was
fetched and checked at `1500ed7`. Work continues on
`wp09/generalized-evaluation`, retaining the complete WP-08 ancestry.

The interrupted finite-pad control evidence was preserved in commit `c8a11a2`.
No failed recording, original camera configuration, scoring threshold or raw
physics state was overwritten. New work is camera rig v4 and Evaluation v2;
see their [camera report](WRIST_CAMERA_V4.md) and
[evaluation report](EVALUATION_V2.md).

## Recovered completed experiments

| Previously outstanding item | Verified artifact and outcome |
| --- | --- |
| Isolated bootstrap and physical runtime smokes | `/mnt/data/ningan/RoboticCuttingSim-wp08-clean-20261029/artifacts/wp08_physical_smoke_001/result.json`: `PASS_REQUESTED_DEPLOYMENT_SCOPE`; both tasks in both actual runtimes, not full cutting acceptance |
| Native additional capsule CUT | `/mnt/data/ningan/RoboticCuttingSim-wp08-regression-20261029/artifacts/native_capsule_v2_001`: 400 steps, execution and unchanged task score PASS |
| Lab additional capsule CUT | `artifacts/wp08_lab_physical_capsule_cut_v3_001`: 400 steps, two 69.12 g parts, unchanged task score PASS |
| Native randomized CUT seed 43 | `.../RoboticCuttingSim-wp08-regression-20261029/artifacts/native_randomized_v2_001`: episode executes and splits; original task FAIL because measured cut-location support error is about 3 mm versus its frozen 2 mm limit |
| Lab randomized CUT seed 43 | `artifacts/wp08_lab_physical_cut_random43_v3_001`: 400 steps, actual split, original cut-location task FAIL retained |
| Lab v4 Cartesian transfer control | `artifacts/wp08_lab_controls_v4_baseline_001`: actual rigid baseline executes; frozen aggregate still `FAIL_OR_INCOMPLETE_RIGID_CONTROLS`; not promoted into a robust grasp claim |

Both original runtimes' fixed CUT/TRIM evidence remains: CUT task metrics pass,
TRIM detaches material but fails strict surface precision. Native's actual
seven-case control matrix passes its frozen engineering screens. Newton
transfer/contact limitations and delayed coupling-energy acceptance remain
unresolved, independently of the new evaluation architecture.

## New completed scope

Franka 1 now stays at the shared normal park, with a passive fixed hand-local
D405. Native and Lab camera-v4 CUT recordings complete 400 steps; separate
motion, RGB-D/FK/timestamp, initial-collider and bracket audits distinguish the
clear secondary mount from the preserved primary bracket/link7 interference.
The same-instance remote viewer has been restarted with v4, remains paused on
Global, and passes actual local Chrome video/input checks. An external Mac
connection is still not claimed.

Evaluation v2 provides a working shared read-only interface, continuous metrics,
independent geometry targets, predeclared ROIs, all raw missing masks, source
and protocol hashes, three profiles and research episode export references.
Historical fixed TRIM, randomized three-bump TRIM and new actual CUT recordings
have been independently reevaluated without changing any original score.
The uncut reconstruction control and adversarial CPU cases retain missing and
invalid data explicitly. The latest full CPU regression passes 414 tests.

After the user's follow-up about the remaining primary-camera slant, v5 gives
both wrist cameras an identity rotation relative to their actual hand. New
Native CUT/TRIM recordings, Lab smoke and the full 401-pose CUT visual-clearance
audit validate this configuration. The old footprint-in-wrist-frusta visibility
screen remains failed and independent of cutting; cameras are not aimed to
pass it. README images/videos and the restarted viewer now use v5.

The README image and videos are produced only from actual camera captures;
old media remain at their original paths. Public compact receipts
link each run to its original local data and hashes. This completes the stated
camera/evaluation delivery, while full WP-08 scientific physics acceptance
remains false. No policy training, system driver or unrelated project was
changed.

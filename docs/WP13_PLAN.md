# WP-13 — Integrated cutting physics

Research payload complete, 2026-11-01 Asia/Seoul. Candidate **REJECTED FOR PROMOTION**;
v5 / Law 8 and `particle_surface_v1` remain the defaults. Branch
`codex/wp13-integrated-cutting`, baseline `11895016b34ed821be2bdecbc5351667fb62d47d`.

| Phase | Result | Evidence / limit |
| --- | --- | --- |
| A Diagnosis | PASS | Reproduced v6 failure; outer corners genuinely stay ahead of the finite edge |
| B Kinematics/front | PARTIAL / promotion REJECTED | CUT gains, partial ligament intact; primary TRIM and held-out cuts regress |
| C Contact | PARTIAL | Two dt values measured; CUT less sensitive, TRIM support overlap worse; not converged |
| D Honest surfaces | MEASURED / PARTIAL | 15 states x two unchanged profiles; missing baseline during F and self intersections NOT_RUN |
| E Focused cohort | 32 recorded / requested-planner integrity FAIL | v5 14/16, v7 9/16 observed; P01 duplicate; Native task/HOME 6/6 |
| F Engineering payload | READY FOR VERIFIED RELEASE | Real 5-GIF/7-MP4 gallery, compact receipts, review and local checks; PR/main CI and browser proof recorded after merge |

The legacy driver dropped compiled overrides. All paired actual paths match,
but 0/32 match all requested planner settings. Completed runs are preserved;
no revised outcomes replace them. Future runs now use compiled options by
default; explicit `legacy_goal_v1` reproduces the deviation. Corrected P01-depth
physics remains NOT_RUN. No renderer cosmetics or further physical alternatives.

[Results and all limitations](reports/WP13_RESULTS.md),
[completed cohort](evidence/wp13/focused_001.json),
[Native evidence](evidence/wp13/native_001.json). Materials remain uncalibrated.
Planning/review may use xhigh; implementation/testing use high. Independent
sequential read-only review receipts, exact-commit CI and actual rendered main
GIF/MP4 checks are release gates. The runner writes publication verification
only after those gates, then DONE last; scientific rejection is not promotion.

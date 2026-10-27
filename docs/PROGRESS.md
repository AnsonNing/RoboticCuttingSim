# Progress

2026-10-28 Asia/Seoul. Active first implementation; final evidence to be added.

| Work package | Current state | Evidence / remaining work |
|---|---|---|
| WP-00 | PARTIAL / BLOCKED | Local isolated repo; source checkouts pinned; hardware audited. Full GA CUDA 13 / Isaac Sim runtime blocked by R570 driver and missing GA dependencies. |
| WP-01 | IN_PROGRESS | Continuous geometry, tray, CSV blade trajectories and Newton MPM implemented. GPU smoke: 19,340 particles, 3 finite frames. Full settling/indent/scrape validation running next. |
| WP-02 | IN_PROGRESS | CRESSim CUDA 12.8 sm_120 shared libraries and headless adapter build successfully. Cutting/withdrawal proof remains open. |
| WP-03 | IN_PROGRESS | ParticleSurface mesh and ideal Warp ray camera implemented; RGB-D execution pending. Isaac RTX / D455 BLOCKED. |
| WP-04–06 | NOT_RUN | Robot integration waits for G2/G3. No policy training. |

PHYSICS_PROVEN=false; VISION_PROVEN=false; WORKCELL_PROVEN=false;
BENCHMARK_READY=false. No particle deletion, presegmented bumps or fake cuts.

User authorized creating and pushing **public** GitHub repository:
https://github.com/AnsonNing/RoboticCuttingSim . This supersedes the original
no-push instruction; no other repository is modified.

# WP-10 handoff to Claude — 2026-10-29

User requested stopping Codex work and letting Claude continue. This is a
checkpoint, not WP-10 completion. Work is on `claude/lucid-mendel-vt7n82` in
`/home/ningan/RoboticCuttingSim`. Last completed package is WP-10B, commit
`02ceca61f04dfb4cd1426effe31f077b202e1f08`; the later handoff commit preserves
partial C/D implementation and experimental drafts. No final PR or merge.
The preserved C/D source checkpoint is `3ce2506`; subsequent wrap-up changes
only add handoff/evidence documentation.

Read `AGENTS.md`, `docs/PROGRESS.md`, `docs/WP10_SPECIFICATION.md`,
`docs/POLICY_CONTROL.md`, ADRs 0003–0005 and reports A–D. The specification's
initial CPU-only section is historical; use the newer reports for GPU status.

| Package | Current evidence | Remaining acceptance work |
| --- | --- | --- |
| B | PASS live interface checks in both runtimes; separate Lab physical-grasp stress FAIL retained | Report complete: [WP-10B](reports/WP-10B.md) |
| C | Native reference CUT-01-R / CUT-02 strict PASS; Lab cuts strict FAIL; Native TRIM strict FAIL; Lab TRIM pregrasp FAIL | Live candidate-scene validation, TRIM/holding diagnosis, actual ruler media; [WP-10C](reports/WP-10C.md) |
| D | GPU law/material probes partly PASS; skim FAIL; graph position comparison FAIL | Exact remaining negative controls, law-1 matrix, full physical-motion criteria; [WP-10D](reports/WP-10D.md) |
| E | Formal matrix NOT_RUN; only scoped TRIM Newton geometry preflight PASS | Both runtimes, all motions/angles/two speeds, full collision and reachability map |
| F | Formal benchmark NOT_RUN; camera-free state and sparse-history paths implemented | Speed/fidelity tables and independent multiple-instance feasibility |
| G | Learned policy, PPO, actual LLM and required annotated robot-video matrix NOT_RUN | Complete every controller/task/runtime requirement in the specification |

## Immediate state and checks

At handoff no WP-10 experiment process remains running. Background viewer
restoration is recorded separately below. Do not stop another user's GPU jobs.
Both 5090s and actual Isaac runtimes worked; there is no GPU-access blocker.
Native is real PhysX/Kit/RTX; Lab is Newton/MJWarp with ideal Warp RGB-D.
Do not present ideal Lab cameras as RTX depth sensing.

Last full CPU run: 461 PASS, 0 SKIP, pytest 92.45 s, outer wall 92.91 s,
`artifacts/wp10/cpu_full_002` (before subsequent scenario/evaluator drafts).
Final checkpoint quick run: **451 PASS, 18 deselected**, pytest 4.66 s, outer
wall 5.02 s, `artifacts/wp10/handoff_cpu_quick_001`. This is not a new full
CPU-physics regression. All source parses and the v3 Warp CPU kernels compile,
`handoff_v3_compile_001`, 3.50 s. **V3 physics remains NOT_RUN.**

`tlmpm.py`, `cutting_tasks_v1.json` and `scene_validation.py` are byte-identical
to the starting commit `1fedbadd3798572769f57dd8c4f85215f686e33d`.
No old experiment directory was overwritten. No policy has been trained.
`gymnasium==1.4.0` was installed locally in `.venv` and the Sim environment;
no drivers or other-project files were modified. Use the correct isolated runtimes:
`.venv/bin/python`, `.deps/isaacsim-venv/bin/python`, `.deps/isaaclab-venv/bin/python`.

## Critical findings to preserve

1. Original seed-42, 400-step Native CUT-01-R and CUT-02 produce two 72 g and
   two 180 g parts respectively. Lab executes both paths without IK rejection
   but retains one component. Native TRIM has 394 rejected IK commands and
   0% bump removal. Lab TRIM never enters the episode: pregrasp angular-speed
   stability fails. A zero shell exit / `PASS_EXECUTION` is not task success.
2. Actual 25-degree TRIM geometry is unsafe at the original table height.
   Versioned scenarios preserve those failures. `ruler_workcell_v2` raises
   the workpiece 50 mm and uses 35 degrees. `c_workcell_v2_preflight_001`
   passes all 401 Newton IK samples, minimum joint margin 0.09146 rad,
   minimum pad clearance 8.27 mm, no penetrating model table/self contacts.
   Its verified Franka-0 seed is saved in `wp10_scenarios_v2.json`.
   **Live execution, dynamic holding and complete collision coverage are NOT_RUN.**
3. Ideal-tracking, full-size 1 mm law-2 TRIM also removes 0%, so reaching the
   path is insufficient. Narrow energized-segment flags and classification
   relative to the tilted blade face leave compatibility bridges.
4. `tlmpm_v3.py` is a newly copied, explicitly optional draft. It classifies
   relative to the surface swept by the moving edge and permits one-sided
   support closure around energized segments. No physical rollout has used
   it. First test side classification, no-energy / side / flat / handle /
   armed withdrawal cases, and laws 1/2 equivalence; then a 1 mm skim.
   Native/Lab presets and current probe CLIs still select law 2; exposing law
   3 through those CLIs/configs is **not implemented**. Only direct explicit
   material/probe construction can select it currently.
5. D's law-summary PASS has scope gaps: withdrawal was disarmed, the handle
   force is 0.271506 N (below the CPU test's 0.5 N assertion), and skim stays
   connected. The graph comparison fails its declared 1 µm tolerance with
   5.375594 µm positional difference despite identical cut codes. Preserve
   these failures. Food dt sweeps are finite-state screens, not convergence.
6. The first reference recorder's tracking error uses held targets after IK
   rejection, so its small TRIM error is misleading as path tracking. New
   records expose `requested_reference_target_world_xyz_xyzw` separately.
7. Franka 1 is a normal parked arm with an actual hand-mounted D405. Keep rig
   v5, fixed hand-local transforms, RGB-D/calibration/time/TF logging, and no
   aiming, observation pose optimization or secondary visibility success gate.

## Raw evidence and source identity

Use [compact evidence](evidence/wp10/wp10cd_handoff_v1.json), plus the existing
[B receipt](evidence/wp10/wp10b_live_validation_v1.json). They contain exact
commands, commit, seed, wall time, result hashes and raw receipt/snapshot paths.
Many C/D commands ran from `02ceca6` with dirty-source snapshots; the commit
alone does not identify the executed code.

Raw artifacts stay on this server under `artifacts/wp10/` and are ignored by
Git. Completed live runs are `c_{native,lab}_{cut01r,trim01r,cut02}_reference_001`.
Use `result.json` and `failure.json` before decoding any episode. Completed
episodes contain synchronous three-camera captures and calibration/transform
metadata, measured robot/tool poses, material histories and stage exports.
Lab TRIM has no completed episode. Command receipts are nested separately in
`c_{native,lab}_batch_001/{cut01r,trim01r,cut02}/command.json`.

Further evidence: `c_{native,lab}_grasp_controls_001`,
`c_{native,lab}_cut01r_clearance_001`, `c_trim_kinematic_formal_001`,
`c_workcell_v2_preflight_001`, `d_{law,materials,graph_check}_gpu_001`.
GPU-probe command receipts are in `d_physics_batch_001/{law,materials,graph_check}`.
Raw per-case NPZ files contain force curves and states. Earlier failed/aborted
pilots remain available; do not count them as completed episodes.

## Suggested resumption commands (not run during wrap-up)

First inspect repository/GPU state, then run the complete CPU suite:

```bash
cd /home/ningan/RoboticCuttingSim
git status --short
git log -5 --oneline
nvidia-smi
.venv/bin/python -m pytest -q
```

After checking that GPU 1 is free, a fresh 35-degree scene smoke can be recorded
with the existing law-2 preset. This is a diagnostic, not expected proof of
TRIM success. Use a new suffix if any output already exists:

```bash
.venv/bin/python scripts/wp10_record.py --out artifacts/wp10/claude_trim_smoke_record_001 -- \
  .deps/isaacsim-venv/bin/python scripts/run_wp10_live.py \
  --runtime isaac_sim --task TRIM-01-R --preset policy --controller reference \
  --scenario ruler_workcell_v2 --ik-iterations 128 --gpu 1 --steps 2 \
  --out artifacts/wp10/claude_trim_smoke_001
```

The Lab equivalent uses `.deps/isaaclab-venv/bin/python --runtime isaac_lab`.
`--grip-force`, `--settle-seconds` and `--grip-friction` are now explicit optional
diagnostic overrides; none of those new override trials has run. Keep the frozen
pregrasp tests and record any new grip choice in an ADR. New failure handling
saves Lab settling/grasp traces where available; earlier failures retain their
original, more limited receipts. `--oracle-stride 10` samples oracle history at
5 Hz while action/robot logging stays at 50 Hz; default stride 1 remains intact.

Continue C → D → E → F → G in specification order. G requires script through
Gym, state BC plus image training/inference, PPO, **actual** observation-driven
LLM tool calls (the old agent-API macro is only a script), and the agentic-control-style
`run(env, ctx)` demonstration within this repo. Every controller/task needs an
actual-robot synchronized RGB-D movie annotated with controller/time/cut/result
in `docs/media/wp10/`. These movies have **not** been delivered. Existing
`docs/media/wp10_kinematic/` movies are supplements only. Complete reports and
README links, push each finished package, then open a PR to main without merging.

## Restored viewer

The previously owned v5 CUT-01 viewer is restored and READY, paused at initial
simulation time, in tmux `rcs-wrist-v5-remote-restored` on GPU 0. The existing
web server remains at `http://100.111.162.17:8210`. Fresh output:
`artifacts/remote_interactive_v5_wp10_restore_001`; startup log:
`artifacts/wp10/viewer_restore_001.log`. See
[restoration receipt](evidence/wp10/viewer_restoration_20261029.json).
Browser/Mac connectivity was not revalidated during wrap-up. If GPU 0 is
needed, stop this specific owned viewer gracefully with
`touch artifacts/remote_interactive_v5_wp10_restore_001/STOP`, then verify its
PID exits. Do not stop unrelated jobs. Original viewer output is preserved.

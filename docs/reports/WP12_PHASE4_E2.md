# WP-12 Phase 4 E2: envelope confirmation through the real planner pipeline

Date: 2026-10-31 (KST). Branch `claude/wp12-physical-cut-surfaces`. Protocol:
[`configs/wp12/phase4_e2_protocol_v1.json`](../../configs/wp12/phase4_e2_protocol_v1.json). Runner:
`scripts/run_wp12_phase4_e2.py` (CPU tests `tests/test_wp12_phase4_e2.py`).

> **Status: protocol written, formal rows NOT_RUN.** Before the commit, only DEV runs were made: a CPU feasibility
> pass over all 20 rows and one GPU control row that is not part of the protocol. They are not evidence. The
> orchestrator commits the protocol, and then the formal rows run.

## 1. Decision follow-up

Source: [WP12_PHASE4_XHIGH_DECISION.md](WP12_PHASE4_XHIGH_DECISION.md), "Next steps", items 1–4.

| Step | Status |
| --- | --- |
| 1. Tests that pinned the TEMPLATE envelope | **Done.** The three tests now check the declared file's invariants. The pending-null behaviour is tested on a template built in the test from the declared rules. `tests/test_wp12_generalization.py`: 38 passed |
| 2a. `freeze_problems` checks `when` rules | **Done.** It also checks every declared ID sample. On the pre-D5 DRAFT it flags the B tilt range and exactly the two solid-board B ID benchmark samples above 9.9° (`s52006`, `s52009`), plus C `trim_layer`. On the re-declared DRAFT it reports nothing. See [WP12_GENERALIZATION.md §5](WP12_GENERALIZATION.md) |
| 2b. Re-declare the DRAFT per D5 | **Done** (still a DRAFT). DEV feasibility `artifacts/wp12/phase5_dev_003`: 26/30 valid_feasible; every ID dev sample is SUPPORTED |
| 3. E2 protocol before any row | **Written** (this file, §2). Not yet committed. The decision calls the file `phase4_e2_envelope_confirmation_protocol_v1.json`; per the user instruction it is `phase4_e2_protocol_v1.json` |
| 4. Resolution erratum | **Done.** Dated section in [WP12_RESOLUTION_STUDY.md](WP12_RESOLUTION_STUDY.md); summarizer fixed (the probe is excluded from fits); new evidence file `phase4_resolution_002.json`, with `_001` kept |
| 4. PROGRESS / README wording, 4c merge | Not in this task. The orchestrator corrected `PROGRESS.md:15` in `fd67c4f` |

## 2. Protocol summary

**Setup.**
- The real pipeline: Phase 5 generators → `scene_config` compile → `goal_planner` with `controller_profile path_v2`
  → `WaypointPolicy` → kinematic ideal tracking → `goal_eval_v2a`.
- Physics: law 8 and `tlmpm_v5`, 1 mm particles, 4 mm grid, 0.1 ms material dt, seed 42, `cutting_tasks_v1`.
- `cut_diagnostics_v1` is on through the read-only wrapper `run_wp12_phase4a_diagnose.py`, with no law parameter
  overridden.
- Budget: ≤ 2 GPU-h, at most 2 concurrent jobs, on cuda:1 / cuda:0 shared with the V1–V4 queue. No Native row
  (decision D5: Demos 2 and 6 go to Native only after their E2 rows).

**Rows.** There are 20, exactly the decision's list. The workpiece is the 160 × 30 × 20 mm bar at the board centre.
The dome is rx 12, ry 10, h 8 mm.

| Row | Cell tested | Prediction (declared) |
| --- | --- | --- |
| SLOT_p20, SLOT_m20, SLOT_p25 | slotted Option D, θ +20 / −20 / +25°, planner 4 mm overtravel (Demo 2 configuration) | CONFIRMED (moderate; +25° low: handle slack) |
| SOLID_p02 | solid board θ 2°, planner armed end | no directional prediction |
| SOLID_p05 | solid board θ 5°, planner armed end (falsification test of D1) | **FAIL** (connected) |
| YAW_p15, YAW_m15 | vertical plane yawed ±15° | CONFIRMED (moderate) |
| ADV_p15, ADV_m15 | slanted chop β ±15° | **NOT_EXECUTABLE_BOARD_COLLISION** (changed after the DEV feasibility, §3) |
| CUT_v08, CUT_v12 | vertical CUT at 8 / 12 mm/s | CONFIRMED (high) |
| SKIM_v20, SKIM_v30 | dome skim at 20 / 30 mm/s | CONFIRMED (high) |
| SKIM_r35 | dome skim ρ 35° | CONFIRMED (moderate) |
| SKIM_psi_p10, SKIM_psi_m10 | dome skim ψ ±10° | CONFIRMED (moderate) |
| TRIM_PLANE | C `trim_plane_protrusion` | CONFIRMED (moderate) |
| TRIM_TRIM | E `trim_trim` | CONFIRMED (moderate) |
| CUT_PUSH | G `cut_push` (sever, kerf_face push 20 mm) | CONFIRMED (moderate) |
| CUT_PUSH_TRIM | Demo 6 candidate: sever → push +x piece → trim the dome on the stationary piece, one episode | CONFIRMED (low; class (c) risk) |

**Verdicts per row.**

- **CONFIRMED** needs every one of:
  - a formal, complete run;
  - the plan finished;
  - a valid solver state;
  - goal_eval_v2a PASS;
  - for sever rows: components ≥ expected and 0 direct code-0 bridges on the sever digit;
  - a motion preflight PASS.
- **PASS_KINEMATIC_PREFLIGHT_FAIL**: the outcome passes, but the preflight fails.
- **FAIL**: an outcome check fails.
- **NOT_EXECUTABLE_BOARD_COLLISION**: the preflight finds the blade or handle inside a board body. The kinematic blade
  does not see the rigid board, so such a row is not executed.
- **ERROR**: a crash or a missing result (kept).
- **NOT_RUN**: budget, infeasible, or DEV only.

**Armed end.** For every armed segment the report gives the height of the lowest edge point above the board top at
the last armed step. It is compared with the 1 mm clean-cut window [0, 0.375] mm.

**SOLID_p05 reading.**
- **FAIL** with an armed end in (0, 0.375] mm supports the D1 armed-end argument.
- **CONFIRMED** falsifies it. Flat-board ≤ 5° in the planner would then become a PARTIALLY_VALIDATED candidate, and
  the xhigh reviewer decides.

**Envelope v2.** The result is a proposal only, `configs/wp12/supported_envelope_v2_PROPOSED.json`, decided by xhigh.
- CONFIRMED makes the tested point VALIDATED.
- FAIL makes it a KNOWN_FAILURE.
- Preflight-only failures make it KINEMATICALLY_UNREACHABLE on this workcell.
- No bound is widened beyond v1.
- A range enters the Phase 5 ID only if both of its tested ends are CONFIRMED.

## 3. DEV runs before the commit (not evidence)

**CPU feasibility of all 20 rows** (`artifacts/wp12/phase4_e2_dev_001/feasibility`, 2.2 min). Each row ran the
generator checks, envelope v1, compile, the staged_v2 motion preflight and the plan facts. No physics was run.

| Finding | Consequence |
| --- | --- |
| **Slotted, planner 4 mm overtravel: knife-handle slack 2.516 mm (+20°), 2.515 mm (−20°), 1.228 mm (+25°).** The planned lowest edge point is 3.6–3.8 mm into the groove. The decision estimated ≈ 2.5 mm at 20°, and this confirms it. The Phase 5 DEV sample at 32.1° collides by 0.68 mm (`phase5_dev_003`) | Rows reachable as planned |
| **Solid-board rows plan the armed end at h = 0.** The armed end is then set by the `path_v2` position tolerance, which the formal rows measure | — |
| **ADV ±15° (slanted chop) is not executable on the solid board with this planner.** `through_cut_rotation` keeps the edge perpendicular to the advance, so the edge line tilts by β. `_relief_end` stops the stroke with the edge **centre** on the board top. The low end of the 120 mm edge is therefore 60 mm × sin 15° = **15.5 mm inside the board** (`blade_collision` −15.53 mm at HOLD:cut0). ADV_m15 also has a negative knife-handle clearance | The predictions changed from CONFIRMED to NOT_EXECUTABLE_BOARD_COLLISION, and the board-collision rule was added. A flat-board slanted chop needs a planner change (stop at the lowest edge point), which is out of scope. D2's "advance tilt β: kinematically reachable not checked on this workcell" becomes **not executable as planned** |
| **CUT_PUSH_TRIM, first draft** (dome on the pushed piece, trim planned with `planner_offset_local_m`): the static preflight checks the unmoved reference bump, giving `blade_collision` −4.85 mm at PRE_CUT:p0 against `static_workpiece_bump` | The dome moved to the stationary piece. The preflight then sees the true geometry |
| **CUT_PUSH_TRIM is physically_unsupported by envelope v1:** the operation pattern `sequence:sever>move_piece>remove_protrusions` is not listed | Run anyway: E2 measures that cell (`verdict_rules.infeasible_rows`) |

**GPU DEV control row** `DEV_CUT_v10` (`artifacts/wp12/phase4_e2_dev_001/DEV_CUT_v10`). This is not a protocol row:
it is the already validated vertical CUT at 10 mm/s, run through the same runner, wrapper and summarizer.
- Exit 0. GNU time 6:43 wall, max RSS 1.26 GB.
- Episode 395 s for 9.5 s simulated: 0.832 s per 20 ms control step on cuda:1, shared at 100 % utilization.
- goal_eval_v2a PASS; 2 components; 0 direct bridges; armed end 0.20 mm above the board top.
- `analyze_row` reads every output.

**Budget consequence.** The per-row estimate is 42 s × episode duration + 30 s, which gives **2.24 GPU-h** for the 18
executable rows. That is over the 2.0 GPU-h cap. The rows are declared in priority order:
1. Demo 2 slotted;
2. the D1 falsification rows;
3. Demo 6, `cut_push`, `trim_plane`, `trim_trim`;
4. plane yaw;
5. speeds;
6. ρ 35° and ψ ±10° last.

At the measured throughput the cap stops at SKIM_r35. So SKIM_r35 and SKIM_psi_±10 are expected to be NOT_RUN
(budget) unless the GPUs are less loaded. Raising the cap is the orchestrator's choice **before** the commit, never
after a result.

# WP-11 review: material-level negative controls (Phase R)

Date: 2026-10-30. Protocol `wp11_phaseR_negative_controls_v1`
([`configs/wp11/phaseR_negative_controls_v1.json`](../../configs/wp11/phaseR_negative_controls_v1.json)).
It was declared, committed and pushed in `18c4301` before any run, smoke or formal. Evidence:
[`docs/evidence/wp11/phaseR_negative_controls_001.json`](../evidence/wp11/phaseR_negative_controls_001.json).

These runs close two review DoD gaps:

- **(a) Phase C gap.** C4 ran its negative controls at ρ = 5° (TRIM-02-K2-R05; the flat press at R00), but the
  live TRIM uses ρ = 10°.
- **(b) Phase M gap.** ADR 0014 validation item 4 says the negative controls hold under `tlmpm_v5`. That was shown
  only by CPU unit tests on a coarse block, never at formal resolution with a multi-digit packed history.

**Label.** Protocol (a) is **POST_LIVE**: it runs after the live TRIM at ρ = 10° (C5/C5b). It does not gate or
re-score any live record.

**Result: 14 / 14 formal rows PASS** (7 in (a), 7 in (b)). There were no failures, errors or invalid controls.
All 14 coarse smoke rows were also PASS_NONFORMAL; these are validation only and are never scored.

## Scope

- Kinematic ideal tracking only: no robot, IK, grasp, slip or cameras.
- Formal resolution: 1 mm particles, 4 mm grid, 0.1 ms material dt, 20 ms control dt. Cutting law 8, seed 42,
  `ruler_knife_v2_y`.
- The material is provisional and uncalibrated (E = 50 kPa, ν = 0.3, yield 1 kPa).
- Oracle material state is used for evaluation only.
- No law parameter, threshold or solver changed.

## How the runs were made

- **Runner.** `scripts/run_phase_r_negative_controls.py` is new. It loads `run_phase_f_material.py`,
  `run_phase_f_v2_material.py` and `run_phase_c4_trim_material.py` (and through them `run_phase1_matrix.py`)
  read-only, as private module instances. In its own instance it registers the extra plans and picks the solver.
  None of their files, rows or records changed.
- **Episode loop.** Phase F's `run_episode`: per-step forces, contact statistics, stroke and label records.
- **Criteria.** Those of the counterpart row, unchanged, with the Phase F contact-window forces. Where C4 used
  an all-steps force mean, it is reported alongside.
- **Protocol (b) scoring.**
  - Every armed waypoint writes digit 4 = K − 1, the top digit and the boundary of the `cut < K` guard.
  - Native packed checks run on the recorded packed codes: encoding packed2 with K = 5, zero slot overflow,
    digits 0–3 never nonzero in any history frame or label record, every armed step reporting `cut_index` 4,
    and digit values ≤ 2.
  - The counterpart's unchanged v4 analysis then runs on an exact single-digit view: codes → digit-4 value,
    stroke 5 → 1, `cut_index` 4 → 0.
  - The view is exact only when no other digit is ever nonzero, which is itself checked. The packed-graph
    component count must equal the view's count (`native_packed_graph_matches_view`).
- **Provenance.**
  - Every row ran from the clean pinned worktree `/mnt/data/ningan/RoboticCuttingSim-wp11-R2-18c4301` at
    `18c4301` (`git_dirty` false), on `cuda:1`.
  - `cuda:1` then had at most 2 other jobs: an unrelated aw0 eval and, at first, the final_v4 regression's
    context.
  - Two lanes ran in parallel, (a) and (b), one row at a time each.
  - `package_file` shows the worktree's `src`.
  - Free space on /mnt/data stayed ≥ 16 GB. The rows take 2.6 GB in total.

Commands (lanes `phaseR_formal_lane1.log`, `phaseR_formal_lane2.log` in `artifacts/wp11/logs/`):

    cd /mnt/data/ningan/RoboticCuttingSim-wp11-R2-18c4301
    TMPDIR=/mnt/data/ningan/tmp /home/ningan/RoboticCuttingSim/.venv/bin/python scripts/run_phase_r_negative_controls.py \
        --row <row> --variant formal --device cuda:1 --out /mnt/data/ningan/RoboticCuttingSim_artifacts/wp11/phaseR/<row>_law8[_v5]_001
    # smoke (non-formal, 2 mm / 8 mm / 0.2 ms): --variant coarse --out …/phaseR/smoke_coarse_<row>_law8[_v5]_001

Outputs: `artifacts/wp11/phaseR/<row>_law8[_v5]_001/` (`manifest.json`, `actions.jsonl`, `physics_state.npz`,
`result.json`); per-row logs `artifacts/wp11/logs/phaseR_<row>_law8[_v5]_001.log`. Evidence was written by
`scripts/summarize_phase_r_negative_controls.py` (read-only, refuses to overwrite).

## (a) C4 negative-control set at ρ = 10° (TRIM-02-K2-R10, tlmpm_v4, K = 1) — POST_LIVE

Declared choices:

- **Handle geometry.** The **phaseF_v2 loaded geometry** (F2.9_blunt_handle_v2 parameters unchanged) on the
  TRIM-02 bar, whose support floor lies directly under the footprint at local x 43–67 mm, y ±11 mm. The C4
  dome sweep was not used: at R05 it failed because the soft dome yields before the grip loads it.
- **Flat press.** The C4 `flat_press` exists only for ρ = 0. At ρ = 10° the applicable flat press is
  `flat_press_normal`: an armed press along the blade-face normal at the task rake. The lowest blade point goes
  from 3 mm above the bar top to 8 mm into it (so the whole face is in), then a 0.3 s armed hold. The window is
  press + press_hold.

| Row | Criterion (counterpart's) | Measured | Verdict | v4 counterpart(s) | Wall s |
| --- | --- | --- | --- | --- | --- |
| Ra_no_contact | zero labels; 0 blade contacts in skim; 0 blade contacts and 0 grip force on armed steps; 1 component; mass | 167 armed steps, 0 blade-contact steps, 0 grip-force steps, max labelled 0, 1 component, mass ✓ | **PASS** | C4 R05 PASS (0 contacts); F2.9_no_contact (R10) PASS | 333 |
| Ra_side_push | zero labels; 1 component; contact-window mean \|F\| > 0.1 N; mass | window mean 0.308 N (max 0.412, 61 of 100 steps in contact; all-steps mean 0.188 N); labels 0; 1 component | **PASS** | C4 R05 PASS (all-steps 0.230 N); F2.9_side_push (CUT-03-K2) 0.318 N PASS | 335 |
| Ra_flat_press_normal | zero labels; 1 component; contact-window mean \|F\| > 1 N; mass | window mean **1.212 N** (max 1.932, 55 of 71 steps; all-steps 0.939; press-only window reported in the evidence); labels 0; 1 component. Declared prediction 1–2 N | **PASS** | C4 R00 FAIL (all-steps 0.931 N); F2.9_flat_press (R00) 1.490 N PASS | 331 |
| Ra_blunt_handle_v2 | zero labels; 1 component; grip contact-window mean > 0.1 N; mass; validity: 0 blade-contact steps, footprint on the bar top, over the floor, off the dome | grip window mean **0.848 N** (max 1.012, 25 of 40 steps); 0 blade-contact steps; footprint x 43–67 mm (dome \|x\| ≤ 12 mm); bar COM moved 0.22 mm, yaw −2.6e-5° | **PASS** (validity all true) | C4 R05 dome sweep FAIL (0.033 N); F2.9_blunt_handle_v2 (CUT-03-K2) 0.855 N PASS | 327 |
| Ra_stop_mid | labels unchanged during the 0.5 s armed stop; resumed outcome report-only; mass | hold change 0 (11338 → 11338, stroke inactive); resumed run PASS_TRIM_SEPARATION (removed 0.979) | **PASS** | C4 R05 PASS (hold 11196, change 0) | 298 |
| Ra_lift_mid | labels unchanged during the armed lift; far region ≥ 99% attached; no label beyond the lead bound | lift change 0; far region (340 particles, x > 2 mm) 100% attached, 0 labelled; max labelled x −0.5 mm ≤ 8 mm bound | **PASS** | C4 R05 PASS (same far/lead numbers) | 295 |
| Ra_half_length | labels unchanged during the stop, the backward withdrawal and the lift; far region and lead bound as lift_mid | hold, withdraw (84 steps) and lift (150 steps) changes all 0; far 100% attached; max labelled x −0.5 mm | **PASS** | C4 R05 PASS | 293 |

## (b) tlmpm_v5, packed K = 5, armed digit 4 — each on its v4 counterpart's task and plan

| Row | Task / plan | Criterion (counterpart's) | Measured (single-digit view) | Native packed checks | Verdict | v4 counterpart | Wall s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rb_side_push | CUT-03-K2, matrix side push | zero labels; 1 component; window > 0.1 N; mass | window **0.31808 N** (64 of 100 steps); labels 0; 1 component | packed2/K5 ✓, overflow 0, max slots 1, single digit ✓ (100 armed steps at index 4) | **PASS** | F2.9_side_push 0.31808 N PASS | 332 |
| Rb_flat_press | TRIM-02-K2-R00, C4 flat press | zero labels; 1 component; window > 1 N; mass | window **1.4900 N** (25 of 40 steps); labels 0 | ✓ (40 armed steps) | **PASS** | F2.9_flat_press 1.4900 N PASS | 339 |
| Rb_no_contact | TRIM-02-K2-R10, C4 no-contact skim | 0 blade contacts and 0 grip force on armed steps; zero labels; mass | 167 armed steps, 0 contacts, 0 labels | ✓ | **PASS** | F2.9_no_contact PASS | 329 |
| Rb_stop_mid | TRIM-02-K2-R05, C4 stop_mid | labels unchanged during the stop; resumed outcome report-only; mass | hold change 0 (11196 → 11196); resumed PASS_TRIM_SEPARATION (removed 0.977) | stroke values {0, 5}; digit-4 labels 36758, digits 0–3 zero; max slots 3; overflow 0; packed graph 2 = view 2 | **PASS** | C4 R05 PASS (hold 11196; final labels 36774) | 337 |
| Rb_withdrawal | CUT-03-K2, half-depth then armed withdrawal | 1 component; core band below the stop 100% code 0; labels unchanged during withdrawal; mass | 1 component; core 2400 particles, code-0 fraction 1.0 (swept-plane fit); withdraw change 0 (13496) | digit-4 labels 13496; max slots 3; overflow 0; packed 1 = view 1 | **PASS** | F2.7_withdrawal_v2 PASS (13496 labels, core 1.0) | 292 |
| Rb_half_depth_cut | CUT-03-K2, half-depth then disarmed retract | 1 component (ligament); core band 100% code 0; labels unchanged during the retract; mass | 1 component; core 2400, code-0 1.0; retract change 0 | digit-4 labels 13496; max slots 3; overflow 0 | **PASS** | F2.7/M4 ligament criteria PASS | 294 |
| Rb_blunt_handle_v2 | CUT-03-K2, phaseF_v2 grip press | zero labels; 1 component; grip window > 0.1 N; mass; validity as F2.9_blunt_handle_v2 | grip window **0.8550 N** (25 of 40 steps); 0 blade-contact steps; footprint inside the right pedestal and the bar top | ✓ | **PASS** (validity all true) | F2.9_blunt_handle_v2 0.8550 N PASS | 287 |

Every row's mass check passed and every row was numerically stable (no NaN/inf, det F > 0).

## Interpretation

- **(a)** At the chosen ρ = 10°, law 8 labels nothing under a spine-first side push, a normal press (1.21 N), a
  grip press (0.85 N) or a skim 3 mm above the dome.
  - No label changes during an armed stop, an armed lift-out or an armed backward withdrawal.
  - No label appears ahead of the lead bound, and the uncut far half stays 100% attached.
  - This matches the C4 R05 controls. The two C4 control-design failures are now passed by the corrected
    designs at ρ = 10°: the all-steps force mean of the flat press and the dome-sweep handle. The C4 records
    themselves stay FAIL.
- **(b)** With a five-digit packed history and the top digit armed, every v4 negative-control criterion holds.
  - The force numbers equal the v4 rows to 5–6 significant figures (side push 0.318079 vs 0.318079 N, flat
    press 1.490002 vs 1.489993 N, grip 0.854978 vs 0.854973 N). The labelled counts of the withdrawal rows are
    identical (13496).
  - The stop row's hold count is identical (11196). Its final count after the resumed skim differs by 16 of
    36774 particles (0.04%), within GPU float32 atomic nondeterminism (Phase M: v4-vs-v4 repeats differ at the
    same level).
  - The label-free rows never activate a stroke. So they exercise the armed `cut_index` 4 path and the
    packed-code bookkeeping, but not digit-4 labelling.
  - The stop, withdrawal and half-depth rows do write digit-4 labels (13496–36758) and keep digits 0–3 at zero.

## Limitations

- Kinematic ideal tracking at formal resolution. These are material-level controls, not robot, grasp or camera
  evidence. The material is uncalibrated.
- One formal attempt per row, seed 42. There is no repeat statistic.
- `flat_press_normal` is a declared new design for ρ ≠ 0. Its 1.21 N contact-window mean clears the 1 N level by
  21%, a smaller margin than F2.9 at ρ = 0 (1.49 N).
- Protocol (b) arms one digit per episode. Interactions between digits (a later digit's control after earlier
  cuts) are covered by Phase M Test B/C and F2.11, not here.
- The single-digit view is exact only for single-digit histories, which the native check enforces. It is not a
  general packed-to-ternary path.

## Commits

| Commit | Content |
| --- | --- |
| `18c4301` | protocol declaration, runner `scripts/run_phase_r_negative_controls.py`, `tests/test_phase_r_negative_controls.py` (pushed before any run) |
| this commit | `scripts/summarize_phase_r_negative_controls.py`, evidence, this report, ADR 0014 validation note |

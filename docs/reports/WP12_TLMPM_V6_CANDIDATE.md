# WP-12 Phase 4: candidate solver `tlmpm_v6` (grid-consistent position update) — build and staged acceptance

Date 2026-10-31 (KST). Branch `claude/wp12-physical-cut-surfaces`. Decision:
[`WP12_PHASE4_PROTOTYPE_DECISION.md`](WP12_PHASE4_PROTOTYPE_DECISION.md) P-A (commit `b05c7f0`). ADR:
[0020 (proposed)](../adr/0020-tlmpm-v6-grid-consistent-position-update.md). Protocol:
`configs/wp12/phase4_gridpos_protocol_v1.json`. Evidence: `docs/evidence/wp12/phase4_gridpos_001.json`.

**Status: REJECTED at S1 (kill set) on 2026-10-31; kept as a recorded negative result.** Building was not adopting. Nothing in production selects `tlmpm_v6`; E2, Phase 5,
the envelope, the benchmark and Demos 1–7 stay on `tlmpm_v5` law 8.

## 1. What was built (Stage 0)

| Item | Path | Note |
| --- | --- | --- |
| Solver | `src/robotic_cutting_sim/tlmpm_v6.py` | `class TLMPM(tlmpm_v5.TLMPM)`, only `step()` overridden; `position_update` `"grid_musl_v1"` (default) or `"flip_v5"` (identity, tests only); `solver_receipt()` |
| Helper | `src/robotic_cutting_sim/packed_solvers.py` | `PACKED_SOLVERS = ("tlmpm_v5", "tlmpm_v6")`, `is_packed_solver`, `packed_solver_class` |
| Opt-in plumbing | `live_material.py`, `kinematic_env.py`, `blade_probe.py` | `solver="tlmpm_v6"` / `solver_version=6` only when passed explicitly; v6 receipt key `tlmpm_v6`; cut-aware surface stays v5-only; scene configs (`scene_config.SOLVERS`) unchanged |
| Tests | `tests/test_tlmpm_v6.py`, `tests/test_wp12_gridpos_protocol.py` | S0 checks, opt-in and default checks, protocol / runner / summarizer logic |
| Runner | `scripts/run_wp12_gridpos.py` | `list`, `run --row --arm`, `queue --stage`; frozen runners executed read-only |
| Summarizer | `scripts/summarize_wp12_gridpos.py` | gates, reported deltas, repeat bands, reopen measure |
| ADR | `docs/adr/0020-tlmpm-v6-grid-consistent-position-update.md` | proposed |

**The rule (exact).** v5 step launch for launch; (1) `x_start <- copy(x)` before `_start_cut`; (2) after
`_deformation`, before `_update_stroke` / `_assign_band` / `_assign_lead`:
`x_p <- x_start_p + dt * sum_{c<8, slot[p,c]>=0} weights[p,c] * new_v[nodes[p,c]*S + slot[p,c]]` (the MUSL field
that also updates F). `v_p` unchanged; `_assign_band` keeps `v_p` in its rear tolerance (declared).

**Frozen, verified by sha256 in the tests:** `tlmpm.py`, `tlmpm_v2.py`, `tlmpm_v4.py` (`cb8985eb…`), `tlmpm_v5.py`
(`a7a05b2f…`), `tlmpm_v5_counterfactual.py` (`c24f8f91…`); the Phase F v1 runner pin (`5aea75d7…`);
`run_wp12_resolution.py` (`a7e34259…`) and `run_wp12_phase4_e1.py` (`8fec3636…`).

### S0 results (CPU, Warp 1.17.0 `cpu`)

**S0: PASS** (CPU, `.venv/bin/python -m pytest`, Warp 1.17.0, device `cpu`).

| Check | Result |
| --- | --- |
| `flip_v5` bitwise tlmpm_v5: x, v, F, Fpi, plastic, progress, work, codes, per-substep reactions and stats, slot diagnostics (legacy tray; board fixture + table plane) | PASS (2 cases) |
| `grid_musl_v1` bitwise the V1–V4 counterfactual `position_update='grid_musl'` (same kernel, same insertion points) | PASS (2 cases) |
| \|x − X − u_grid\| < 1e-6 m (`flip_drift_v1`) on a law-8 CPU cut; drift still accrues in v_p; positions differ from v5 | PASS |
| CPU controls under grid_musl_v1: side push, flat press, blunt handle, no contact → 0 labels, 0 severed bonds, 1 component, force present; armed stop and armed withdrawal → labels unchanged, 1 component; half cut → code-0 ligament below the stop | PASS |
| Free block (no supports, no gravity, 2 mm lattice): per substep \|ΔΣmv − blade impulse\| < 1e-6 × total impulse, v5 and v6 | PASS |
| Options, receipts (tlmpm_v5.py sha `a7a05b2f…`), never a default (KinematicConfig, LiveCuttingMaterial, `scene_config.SOLVERS`, no config names v6 except the protocol, only 5 src files name it), kinematic env v6 opt-in and v5 path unchanged (bed class, receipt keys) | PASS |
| Frozen solver files (sha256) and frozen runners (Phase F v1 pin, resolution, E1) unchanged | PASS |
| Protocol / runner / summarizer logic (`tests/test_wp12_gridpos_protocol.py`) | PASS (6) |
| Full CPU suite, run in 16 file shards (`pytest -q -p no:cacheprovider <files>`) | 1583 passed, 5 xfailed, 1 failed → fixed → shard re-run 118 passed. The failure was mine: the first protocol draft named the counterfactual module, which `tests/test_root_cause_recorders.py` forbids in configs; the protocol now describes it without its module name |

**Finding made while building S0 (pre-existing, not a v6 effect).** On the 4 mm CPU test block
(`blade_probe.block`, s = dx) the particles sit on node planes. Corners with zero weight still receive the
stress-gradient term in `_p2g`, and `_grid_update` drops nodes with mass ≤ 1e-15, so total momentum is not conserved
on that lattice: a 0.2 s side push gave ΔΣmv 7e-4 N·s against a 0.0376 N·s blade impulse, in v5 and v6 alike. Formal
1 mm lattices sit at node fractions 1/8, 3/8, … and have no zero-weight corners. The S0 momentum check therefore uses
a 2 mm lattice (fractions 1/4, 3/4), where both solvers conserve momentum to float32 rounding. Nothing was changed in
v5.

## 2. Protocol (`phase4_gridpos_protocol_v1`)

Written from the decision's skeleton before any formal GPU row (the runner refuses formal rows unless the file is
committed and unmodified). 82 rows: S1 10 (+3 repeats), S2 41, S3 29, S4 2 (declared, not scheduled).

- **Arms.** Every scheduled row runs as `V6` (tlmpm_v6 grid_musl_v1, law 8) and `V5` (tlmpm_v5 law 8, same commit).
  Exception: the six `RES_*` characterization rows use the declared resolution_002 records as their V5 arm (same
  `tlmpm_v5.py` and `run_wp12_resolution.py` hashes and row config; the summarizer re-checks and recomputes their link
  metrics). M1a, M3b and FB_CUT run twice per arm (`_r2`) for the GPU nondeterminism band.
- **Solver selection.** In the row process only: `LiveCuttingMaterial.__init__` is wrapped so every material is built
  with the arm solver (the frozen driver asked for None = tlmpm_v4 by law, or "tlmpm_v5"); the `res` and `probe`
  drivers, which import `tlmpm_v5.TLMPM` directly, get the arm class under that name (the V1–V4 mechanism). Every
  redirect, bed class and v6 receipt is recorded; a row whose selection is not verified is INVALID. Frozen runners still
  write their hard-coded solver label into their own `run_*/result.json`; `gridpos_result.json` is authoritative.
  Consequence: the V5 arm of rows that WP-11 ran on tlmpm_v4 (Phase F, M9) is a new tlmpm_v5 record (K = 1 packed
  codes equal ternary codes); the v4 records stand.
- **Monitor (`gridpos_monitor_v1`, read-only).** Per control step: finite, min det F, max speed, labels per digit,
  blade / total force, tool work, blade / fixture overlap, particle KE, min particle-centre z on the solid flat board;
  every 5 steps grid KE (MUSL field) and null-space KE; every 10 steps Hencky elastic energy; FINAL components, mass
  fractions and piece COM velocities; `flip_drift_v1` grid-position residual. `cut_diagnostics_v1` is not attached
  (declared deviation: it brackets `tlmpm_v5.TLMPM.step`, which v6 overrides); E1's own recorders are bracketed for v6
  by the runner so its analysis stays complete.
- **S1 kill rule.** A S1 row FAILs iff its V6 arm (either repeat) does not PASS its frozen criterion, fails integrity
  (finite, min det F > 0.5, slot overflow 0, no centre below the solid board top, solver verified), differs from V5 in
  FINAL component count on a separating row, misses TRIM removal ≥ 0.9 / substrate ≥ 0.99, or errors. A V5-arm failure
  is recorded and never excuses a V6 failure. The queue stops launching on the first S1 FAIL.
- **S2/S3 gates.** Frozen criteria PASS on acceptance rows, no verdict worse than V5, equal components on separating
  rows, TRIM thresholds, integrity. Reported only: force, work, overlap, KE, elastic energy, piece velocities, labels,
  known-failure rows (TEST_D2 s2501, TRIM_W s2307, 4c G07), characterization rows (F3 dt_half, RES_*, DT_*), M8a,
  M9c/d, F2.9_blunt_handle v1.
- **New post-cut controls (ADR 0014 controls, NOT_RUN until now).** CUT-01-R, K = 2, 16 s episode (the 8 s task is
  too short for two actions; the duration is the only task change): a full M1 through-cut writing digit 0, then digit 1
  armed for a flat press across the kerf (M5b geometry), a side push (M5a geometry), or an armed 0.5 s stop at the
  bottom of the kerf followed by an armed withdrawal. PASS iff 0 digit-1 labels, digit 0 unchanged, component count
  unchanged across the digit-1 segment, and the cut separated first.
- **Predictions** (copied from the decision): acceptance rows and controls PASS; q50 ≥ 2 and q99 ≥ 2 on EFF CUT, TRIM,
  MULTI2, S7, G3b; at 0.5 mm CUT q50 < 2; CUT and MULTI2 force and work above v5, OBL20 below; overlap lower;
  known-failure rows unchanged. Risk rows without a confident prediction: M8b, M2, E1_A_p05. dt pair: none.
- **Budget.** Estimate S1 2.3, S2 8.9, S3 5.1 GPU-h (row estimates × arms × repeats); cap 18 GPU-h; rows beyond the
  cap are NOT_RUN. S4 Native (≈ 1 h) is not scheduled: it is needed only before any Native use of v6.
- **Outputs.** `artifacts/wp12/gridpos_NNN/<arm>/<row>/` (`gridpos_result.json`, `monitor.npz`, the frozen runner's
  `run_*` directory), logs `artifacts/wp12/logs/gridpos_NNN_<arm>_<row>.log`, queue ledger
  `artifacts/wp12/gridpos_NNN/queue_ledger.jsonl`. The decision skeleton named the report `WP12_PHASE4_GRIDPOS.md`
  and the runs `phase4_gridpos_NNN`; the orchestrator's names (`WP12_TLMPM_V6_CANDIDATE.md`, `gridpos_NNN`) are used.

## 3. DEV smokes (never evidence)

DEV rows only (label DEV; `artifacts/wp12/gridpos_dev_001`, `gridpos_dev_002`; logs
`artifacts/wp12/logs/gridpos_dev_00N_<arm>_<row>.log`), seed 42, cuda:0 / cuda:1, run before the protocol commit
to exercise every driver kind. Most are capped with `--dev-max-steps` (N control steps), so their verdicts are
meaningless (`FAIL` / `FAIL_NONFORMAL` simply means "plan not finished"). What they check: the driver runs end to end,
the solver selection is verified (redirect or substitution recorded, bed module = arm solver, V6
`position_update = grid_musl_v1`), the monitor and integrity checks fill in, and the frozen runner writes its record.

| Kind | Rows (arm, cap) | Outcome |
| --- | --- | --- |
| matrix | M4 (V6, V5, 30), M1a, M1b (V6, 20), M8b (V6, 20, K = 2) | complete, selection verified, integrity PASS |
| matrix replay | M9a (V6, full 400 steps, in-process replay) | PASS (frozen strict), 2 components, 157 s |
| matrix_m7 | M7 (V6, from the capped M1a/M1b) | mechanics run (FAIL: the sources were capped) |
| phase_f | F2.7_withdrawal_v2 (V6, 20), F3_cut dt_half (V6, 10) | complete (`FAIL_NONFORMAL`, `CHARACTERIZATION`) |
| phase_f | F2.11_multicut_history (V6, 20) | ERROR inside the frozen Phase F evaluator: `decode_digits` on an empty float label-record array, because 20 steps assign no label. A DEV-cap artefact (the formal v5 record of this row passes with the same code); kept |
| phase_f_v2 | F2.9_blunt_handle_v2 (V6, 20) | complete |
| phase_r | Rb_withdrawal (V6, 20) | complete (K = 5 packed, digit 4) |
| workcell | FB_CUT, FB_partial (V6, 20), FB_TRIM_r10 (V5, 20) | complete; solid-board centre check active (min centre z 0.4 mm); TRIM numbers extracted from the goal evaluator |
| postcut | POSTCUT_stop_withdraw, POSTCUT_press (V6, full 800 steps) | **PASS** both: 0 digit-1 labels, digit 0 unchanged (28 500 labels), 2 components before and after the digit-1 segment, plan finished; press peak force 3.7 N. The first stop_withdraw attempt (`gridpos_dev_001`) is kept as ERROR: lengthening the task needs the scripted reference knots extended too (fixed: a final hold knot) |
| verify/res | EFF_CUT (V6, V5, 20–30), RES_CUT_s050 (V6, 3) | complete; V6 uses the declared `tlmpm_v5.TLMPM` substitution, V5 none |
| verify/e1 | E1_A_p10 (V6, 30), E1_D_p20_reach (V5, 20) | complete; E1's own recorders bracketed for V6 (recorder overflow 0); `FAIL_INTEGRITY` = episode incomplete |
| verify/probe | EFF_S7, EFF_G3b (V6, 20–30) | complete |
| phase4c | 4c_G07 (V6, full) | FAIL, as the v5 record (known) |
| generalization | TRIM_W_s2307 (V6, 20), TEST_D2_s2501 (V6, 10) | complete (partial smoke class) |

**Grid-position residual on GPU rows (reported, not gated).** \|x − X − u_grid\| on V6 is 1e-7–2e-6 m on the
short capped rows, 1.5e-5 m on 4c_G07, 5.5e-5 m on M9a (400 steps) and 2–3e-4 m on the 800-step post-cut rows. The
CPU S0 check (< 1e-6 m) holds; on long GPU episodes the residual is consistent with float32 stagnation of x: at
\|x\| ≈ 0.08 m one ulp is 7.5e-9 m, so a particle creeping slower than about 37 µm/s (dt v < half an ulp at
dt = 0.1 ms) never moves while the recorder still integrates its grid displacement. That affects v5 in the same way
(x += dt v_p); it was not verified particle by particle. On V5 the same quantity is the FLIP drift itself
(1e-4 m after 30 steps of M4).

## 4. Stage results

Protocol committed at `28c8379` (guard-test follow-up `c279a4d`). Seed 42, kinematic ideal tracking, 1 mm / 4 mm /
0.1 ms, two lanes (cuda:0, cuda:1), runs `artifacts/wp12/gridpos_001/<arm>/<row>`, ledger
`artifacts/wp12/gridpos_001/queue_ledger.jsonl`, logs `artifacts/wp12/logs/gridpos_001_<arm>_<row>.log`.

```
TMPDIR=/mnt/data/ningan/tmp nohup .venv/bin/python scripts/run_wp12_gridpos.py queue --stage S1 \
    --out-root artifacts/wp12/gridpos_001 >> artifacts/wp12/logs/gridpos_001_queue.log 2>&1 &
.venv/bin/python scripts/summarize_wp12_gridpos.py --root artifacts/wp12/gridpos_001 \
    --evidence docs/evidence/wp12/phase4_gridpos_001.json --extra <notes + FB_CUT diagnosis>
```

| Stage | Verdict |
| --- | --- |
| S0 (CPU) | PASS |
| S1 (kill set) | **FAIL** (FB_CUT) → **v6 REJECTED** |
| S2, S3 | NOT_RUN (the protocol stops on any S1 FAIL) |
| S4 Native | NOT_RUN (not scheduled) |
| Phase 3 reopen measure | NOT_ELIGIBLE (not computed) |

GPU time: 0.93 GPU-h in the S1 queue (ledger walls, 30 s polling resolution) + 0.29 GPU-h of DEV smokes before the
commit. Cap 18 GPU-h.

### S1 rows (V6 vs V5 at the same commit)

| Row | V6 | V5 | Components V6 / V5 | Blade force mean \|F\| (N) V6 / V5 | Tool work (mJ) V6 / V5 | Max blade overlap (mm) V6 / V5 | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1a | PASS | PASS | 2 / 2 | 0.027 / 0.016 | 0.66 / 0.41 | 0.27 / 0.49 | ok |
| M3b | PASS | PASS | 2 / 2 | 0.040 / 0.034 | 1.52 / 1.27 | 0.37 / 1.20 | ok (removal 1.00 / 0.97, substrate 1.00) |
| M4 | PASS | PASS | 1 / 1 | 0.014 / 0.017 | 0.22 / 0.29 | 0.27 / 0.25 | ok |
| M5a | PASS | PASS | 1 / 1 | 0.21 / 0.14 | 4.0 / 4.1 | 0.21 / 0.20 | ok (0 labels) |
| M5b | PASS | PASS | 1 / 1 | 3.17 / 2.94 | 14.8 / 14.1 | 0.72 / 0.62 | ok (0 labels) |
| M8b (risk row) | PASS | PASS | 3 / 3 | 0.050 / 0.037 | 3.7 / 2.9 | 0.25 / 0.54 | ok |
| F2.7_withdrawal_v2 | PASS | PASS | 1 / 1 | 0.014 / 0.017 | 0.22 / 0.29 | 0.27 / 0.25 | ok |
| F2.9_blunt_handle_v2 | PASS | PASS | 1 / 1 | — (no blade contact) | 0 / 0 | 0 / 0 | ok (0 labels) |
| **FB_CUT** | **FAIL** | PASS | **1 / 2** | 0.025 / 0.010 | 0.54 / 0.25 | 0.25 / 0.48 | **KILL** |
| FB_TRIM_r10 | NOT_RUN | NOT_RUN | | | | | stopped |
| M1a_r2, M3b_r2, FB_CUT_r2 | NOT_RUN | NOT_RUN | | | | | stopped |

Integrity passed on every completed row in both arms: finite states, min det F 0.85–0.97, slot overflow 0, solver
selection verified, and on FB_CUT no particle centre below the board top (min 0.36 mm V6, 0.37 mm V5). The
V5 arm also fills in law-8 rows on tlmpm_v5 that had never been rerun (M3b, M4, M5a, M5b, and the Phase F rows
F2.7_withdrawal_v2 and F2.9_blunt_handle_v2, all PASS). This is new v5 evidence; the frozen v4 records are unchanged.

**The kill row, FB_CUT.** This is the Phase 2b CUT episode (`CFG-bar_flat_board-sever_mid-c32f94a4`, path_v2) on the
kinematic backend. On V6 the plan finished and the evaluator's integrity layer passed, but the outcome failed:
`expected_substantial_pieces`, `every_plane_has_an_interface` and `cut_at_declared_planes`, with 1 mechanical
component. The class is `execution_failure` (outcome). On V5 the same episode is `task_success` with 2 components.
A read-only diagnosis (`artifacts/wp12/gridpos_001/FB_CUT_bridge_diagnosis.json`; reference stencil plus
`cut_diagnostics.direct_bridges` on the FINAL packed codes) gives:

- V6 leaves **12** particles unassigned. They are in the bottom layer (reference z = 0.5 mm), within ±1.5 mm of the
  plane, at the two side faces (y = ±14.5 mm). These particles are direct code-0 bridges between the two sides.
- V5 labels every particle within 4 mm of the plane and has 0 bridges.
- V6 labels more of the bottom layer overall (1368 vs 1272 particles) but misses these side-face corner particles.

This is the bottom-layer exit region of a flat-board through-cut where the edge ends on the board top. That is the
same region as the known law-4-front difficulty at the board. Here it becomes a failure on an in-plane vertical cut
that v5 separates.

**Reading.**
- The kill rule is applied as declared: one S1 row FAILs, so v6 is REJECTED. Nothing was tuned or re-run, and every
  output is kept.
- The other 8 kill rows behave as predicted. Topology is unchanged. Blade force and work are mostly higher (M1a +64 %
  force, ×1.6 work), and blade overlap is lower on the cutting rows (M1a 0.49 → 0.27 mm, M3b 1.20 → 0.37 mm).
- FB_CUT was not one of the declared risk rows (M8b, M2, E1_A_p05). M8b passed.
- The declared repeat band was never measured because the repeats were stopped. So whether V5 could also leave a
  corner bridge on another GPU run of FB_CUT is unknown. The kill rule does not depend on this, but a reviewer may
  weigh it.

**Wording.** "tlmpm_v6 (grid-consistent position update) was built and rejected at its kill stage: on the flat-board
CUT it left 12 bottom-corner particles unassigned and the bar did not separate, where tlmpm_v5 separates." Do not
claim a cutting improvement or regression beyond this row. Completion Criterion 4 stays bounded (decision P-A: "If v6
is rejected … report criterion 4 as bounded"). Production, E2, Phase 5 and the demos are unaffected; they never
used v6.

## 5. What must not be claimed

Better or cleaner cutting; a fixed law-8 failure region or an oblique gain; force convergence; fracture;
calibration; "the standard TLMPM update"; full x/F consistency (label timing and the CUT 0.5 mm P50 remain); a surface
gain or "Phase 3 passed" before a v3 protocol; any V1.4 (counterfactual) number as v6 evidence. v6 is never selected,
accepted or tuned on a surface metric.

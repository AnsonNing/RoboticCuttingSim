# WP-12 Phase 2 — `blade_master_flat_board_v1` workcell (draft, part 1)

Date: 2026-10-30. Branch `claude/wp12-physical-cut-surfaces`. Decision: [ADR 0017](../adr/0017-blade-master-flat-board-workcell.md).
Evidence: [`phase2_workcell_001.json`](../evidence/wp12/phase2_workcell_001.json) (geometry, layout preflight,
Native) and [`phase2_throughcut_001.json`](../evidence/wp12/phase2_throughcut_001.json) (kinematic through-cut study).
Material parameters remain provisional and uncalibrated. Nothing in `tlmpm*.py`, law 8, `final_v4`, WP-11 records or
any v1–v4 threshold was changed.

## 1. What changed

| Layer | `wp11_workcell_v1` (kept, unchanged) | `blade_master_flat_board_v1` (new, opt-in) |
| --- | --- | --- |
| Supports | Each support segment: a wood board (10 mm) plus a grey riser down to the table | ONE continuous board (360 × 190 × 20 mm) lying on the table |
| Workpiece bottom | World z = 0.10 m (+100 mm) | World z = 0.01 m (board top); validated, never floating or inside the board |
| Table | 1.8 × 1.2 m, top at z = −0.01 m, centred at the origin | Same top (the Franka mounting plane); y from −0.075 to 0.6 m so the hand can hang past the front edge |
| HOME | q_home (C2), edge at local (0, 0, 0.188) m | New q_home by the runtime IK, same local edge pose (world z 0.198 m) |
| Action workspace floor | Local z ≥ −12 mm | Local z ≥ 0 (board top); ≥ −slot depth for the explicit slotted variant |
| Preflight blade vs board | 1 mm clearance in every sample | 1 mm in free space; contact samples may touch (distance ≥ 0), never enter |
| Slotted board | Gaps between separate boards | Only when declared: through-width grooves in one continuous board |

Scenes: `configs/wp12/scenes/{bar_flat_board, bar_flat_board_slotted, trim01_flat_board, bar_three_bumps_flat_board}.json`.
Goals: `configs/wp12/goals/{divide_5_slices, remove_all_protrusions_rho10}.json`; the wp11 goals `sever_mid` and
`trim_bump_rho10` are reused.

## 2. Scene geometry

Side view at the cut plane (world x = 0), not to scale. The knife edge line is along +y, with the grip at −y:

```
                          z (m)
   hand + D405 hang  ...  0.198  HOME edge (hand straight down, blade hanging)
   past the table edge
          |               0.030 ... 0.038  bar top / dome top (local 0.020 / 0.028)
          |     +----------------+
          |     |   workpiece    |  160 x 30 x 20 mm, bottom = board top
   ======= +====+================+=========================+   0.010  board top  (local z = 0)
   |   (overhang)|      cutting board 360 x 190 x 20 mm     |
   |  y=-0.075  +------------------------------------------+  -0.010  board bottom = table top
   +=========================================================================+
   |   table top z = -0.010 (Franka bases at (-/+0.46, 0.12, -0.01) stand on it)   |
   +=========================================================================+  -0.060
   y: -0.075 table edge, -0.065 board edge, -0.015..+0.015 bar, +0.125 board back, +0.6 table back
```

There is one source for all of it, `configs/wp12_workcells_v1.json` → `flat_board_workcell.resolve`:

| Consumer | Built from | Code |
| --- | --- | --- |
| TLMPM fixture contact | `fixture_boxes_local` (board box in workpiece-local coordinates) | `scene_config.raw_support_boxes` → `ConfiguredTaskSpec.fixture_boxes_local` → `LiveCuttingMaterial` |
| PhysX board collider and visual | The same boxes; one Cube per box, scale = half extents, `CollisionAPI` on that Cube | `isaac_sim_runner._author_task_fixture` (unchanged code path) |
| PhysX table collider and visual | `table_box_world` | `_author_task_fixture` (opt-in branch) |
| Planner support (through-cut end) | `raw_support_boxes` | `goal_planner._relief_end` (unchanged) |
| Collision preflight | `task.fixture_boxes_local` and `task.table_box_world` | C2 `GeometryModel` (opt-in table), `motion_preflight` |
| HOME and action workspace | The record | `scene_config.workcell_home` / `workcell_action_workspace` |

The CPU tests (§7) check that these all agree, including after a thickness or position change.

## 3. Why WP-11 raised the workpiece +100 mm, and the layout chosen here

C2 (ADR 0013) chose +100 mm because of the hand, not the reach. With the end-held `ruler_knife_v2_y`, the edge line is
the approach axis. At ρ = 10° the finger-closing axis is nearly vertical, and the 200 mm Franka hand hangs below the
trim plane. At +50 mm the hand capsule was 16.5–17.3 mm inside the table.

Layout study: `artifacts/wp12/workcell_002`. Command:
`.venv/bin/python scripts/preflight_wp12_flat_board.py --out artifacts/wp12/workcell_002`. It uses the WP-11 motion
preflight (runtime Newton IK plus C2 MuJoCo distances) from the workcell HOME. Test C uses the staged per-protrusion
hulls (`staged_v2`). Geometry only; deterministic; 250 s. Lowest points of Franka 0 relative to the board top:

| Task | Hand | Link 6/7 | Wrist D405 | Finger | Pad | Blade |
| --- | --- | --- | --- | --- | --- | --- |
| CUT (sever_mid) | −12.0 mm (y −0.204…−0.124) | −16.2 mm (y −0.382…−0.249) | +62.2 | +17.5 | +19.3 | 0.0 |
| TRIM ρ = 10° | −80.4 mm (y −0.204…−0.124) | −64.3 mm (y −0.249…−0.194) | −30.4 mm (y −0.197…−0.174) | −13.7 mm (y −0.135…−0.081) | −2.4 mm (y −0.098…−0.082) | +20.0 |

Everything below the board top is on the grip side, at world y < −0.08 m. The layout therefore:

- keeps the table top at the Franka mounting plane, so the bases, HOME family and cameras stay where they were;
- puts the board front edge at y = −0.065 m and the table front edge at −0.075 m, so the hand hangs past the table
  edge during skims;
- keeps the same workpiece-relative HOME as wp11.

Verdicts (PASS = every C2 check passes on the dense IK chain):

| Task | Default layout | Full table (front edge −0.6 m) | Board 30 mm (workpiece raised with it) |
| --- | --- | --- | --- |
| CUT-01-R-like sever | PASS (table clearance 60.8 mm, joint margin 0.618 rad) | FAIL: link 6 3.8 mm from table (< 5 mm) | PASS |
| TRIM ρ = 10° | PASS (table 48.7 mm, pad 16.4 mm from board) | FAIL: hand −56.5 mm, D405 −10.4 mm into table | PASS |
| Test B 4 slices, solid board | PASS (blade-board 0.3 µm at HOLD) | FAIL: link 6 3.8 mm | PASS |
| Test B 4 slices, slotted variant | PASS (blade in groove, 3.25 mm from groove walls/floor) | FAIL: link 6 −0.2 mm | PASS |
| Test C 3 bumps ρ = 10° | PASS (staged_v2) | FAIL: hand −56.5 mm | PASS |

The first run (`workcell_001`) is kept. In it, Test C failed blade vs bump by −7.95 mm at `PRE_CUT:p1`. That is the
known C2 single-hull artefact (one hull over three bumps bridges the gaps; WP-11 Phase E round 1, rule 2), not a
workcell problem. `workcell_002` uses the staged preflight.

IK residuals are ≤ 3.3 µm and ≤ 1.4 µrad. The smallest joint-limit margin is 0.547 rad, so the reach is comfortable
at board level.

## 4. Contact architecture

| Pair | Solver | Friction | Notes |
| --- | --- | --- | --- |
| Material ↔ board (and table plane) | TLMPM fixture contact (grid boundary) | Task material friction (0.3) | Same boxes as PhysX |
| Knife, hand, robot ↔ board and table | PhysX | Record value (`task_material` = 0.3) | The tool meshes have contact offset 0.2 mm and rest offset 0 |
| Knife ↔ material | TLMPM blade coupling (law 8) | — | Reaction is applied to the knife body |
| Workpiece in PhysX | — | — | The material is not a PhysX body, so nothing is counted twice |

TLMPM blade vs board: see §5. The TLMPM blade is kinematic and does not know about the board.

## 5. Through-cut versus solid board (spec §2.4)

**Setup.** Kinematic ideal tracking, `tlmpm_v5`, law 8, 1 mm particles, 4 mm grid, 0.1 ms, seed 42, cuda:0.
- Workpiece: the CUT-01-R bar (cutting_tasks_v1, provisional) and `ruler_knife_v2_y`.
- Motion: a single vertical through-cut at x = 0, a hold, a withdraw and a settle.
- Physics: no edits. The study is a new script, `scripts/run_wp12_board_throughcut.py`; its summary script is
  `scripts/summarize_wp12_board_throughcut.py`.
- Runs: `artifacts/wp12/workcell_throughcut_001…010`, 400–553 s each.
- Evidence and figure: `docs/evidence/wp12/phase2_throughcut_001.json`,
  `docs/media/wp12/phase2_throughcut_bottom_layers_001.png`.
- The lowest point of the vertical TLMPM blade polygon is its edge vertex; this is checked per row.

**Columns.**
- *L0/L1/L2*: lattice links crossing x = 0 that carry conflicting side digits, out of 30, in particle layers 0/1/2.
- *Bridges*: code-0 particles connecting both sides.
- *Below top*: particle centres below the board top.
- *Fz*: material–board force during the hold, from the material momentum balance. The bar weighs 1.413 N.

| row | h (mm) | comp hold/settle | mass fractions (settle) | L0 conflict/links | L1 | L2 | bridges | min z (mm) | below top | Fz hold mean/max (N) | min detF | wall (s) | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S_P100 | +1.000 | 1/1 | 1.000 | 24/30 | 30/30 | 30/30 | 172 | 0.378 | 0 | 1.41/1.41 | 0.957 | 552 | FAIL |
| S_P050 | +0.500 | 1/1 | 1.000 | 28/30 | 30/30 | 30/30 | 16 | 0.378 | 0 | 1.41/1.41 | 0.957 | 553 | FAIL |
| S_P0375 | +0.375 | 2/2 | 0.500, 0.500 | 30/30 | 30/30 | 30/30 | 0 | 0.379 | 0 | 1.41/1.41 | 0.957 | 540 | PASS |
| S_P025 | +0.250 | 2/2 | 0.500, 0.500 | 30/30 | 30/30 | 30/30 | 0 | 0.378 | 0 | 1.41/1.41 | 0.957 | 551 | PASS |
| S_000 | +0.000 | 2/2 | 0.500, 0.500 | 30/30 | 30/30 | 30/30 | 0 | 0.377 | 0 | 1.41/1.41 | 0.957 | 525 | PASS |
| S_M025_DIAG DIAG | -0.250 | 2/2 | 0.500, 0.500 | 30/30 | 30/30 | 30/30 | 0 | 0.377 | 0 | 1.41/1.41 | 0.957 | 532 | DIAGNOSTIC |
| D8_M400 | -4.000 | 2/2 | 0.500, 0.500 | 30/30 | 30/30 | 30/30 | 0 | 0.377 | 0 | 1.41/1.41 | 0.957 | 532 | PASS |
| D8_000 | +0.000 | 2/2 | 0.500, 0.500 | 30/30 | 30/30 | 30/30 | 0 | 0.367 | 0 | 1.41/1.41 | 0.957 | 408 | PASS |
| D40_M400 | -4.000 | 2/2 | 0.500, 0.500 | 30/30 | 30/30 | 30/30 | 0 | -0.651 | 808 | 1.41/1.42 | 0.941 | 403 | PASS |
| D40_000 | +0.000 | 1/1 | 1.000 | 24/30 | 30/30 | 30/30 | 48 | -0.894 | 1088 | 1.41/1.42 | 0.940 | 403 | FAIL |

S_* rows use the solid board, D8_* an explicit 8 mm slot at the cut and D40_* the CUT-01-R 40 mm gap. h is the edge's
lowest point minus the board top. D*_M400 rows go 4 mm below the workpiece bottom, as the existing slotted plans do.
`S_M025_DIAG` is **DIAGNOSTIC_PENETRATION_NOT_ADOPTABLE**: the edge is 0.25 mm inside the rigid board. It is never a
candidate.

![bottom layers per row](../media/wp12/phase2_throughcut_bottom_layers_001.png)

**Findings.**
- **The solid board cuts cleanly with the edge on the board top.**
  - At h = 0, +0.25 and +0.375 mm: two persistent pieces (0.500/0.500), all 30 layer-0 links conflicting, 0 bridges.
  - At h = +0.5 mm, 16 layer-0 particles stay unassigned and the bar remains one component.
  - At h = +1.0 mm, 172 stay unassigned.
  - Only layer 0 is ever bridged.
  - Penetrating the board (h = −0.25 mm) gives exactly the same topology as h = 0, so it would gain nothing.
- **The margin above the board is material-dependent.**
  - Bottom-layer particles rest with their centres 0.43 mm above the board (contact radius).
  - Near the cut they lift to 0.80–0.92 mm, because the wedge spreads the soft halves.
  - The law-4 band used by law 8 labels particles by their current position, so part of the +0.375 mm margin comes
    from this lift. It is not a robust tolerance: re-measure it for other materials, spacings and live tracking.
- **Material–board contact is real.** No particle centre ends inside the board on the solid or the 8 mm board. The
  penalty-shell overlap is 72–101 µm, and the hold force (1.41 N) equals the weight.
- **The 40 mm gap (CUT-01-R semantics) needs the overtravel.**
  - At h = 0 the unsupported bar sags 0.65–0.9 mm into the gap and 48 bridges remain (FAIL).
  - At h = −4 mm it separates (the M1a equivalent). Its "below top" count is material sagging into the open gap,
    not into a board.
- **The 8 mm slotted variant** separates at both h = −4 mm and h = 0.

**What the TLMPM blade does at the board** (code reading at `dd1538b`, file:line in the evidence):
- TLMPM does not know that the board is rigid for the blade. The kinematic blade is a prescribed SDF that acts only on
  material particles (`tlmpm_v5.py` `_rigid_contacts` :185–296). Nothing compares the blade pose with a fixture box,
  so nothing stops, resists or reports a blade entering the board.
- Fixture boxes and the table plane act only on material particles, through a per-particle penalty impulse with
  Coulomb friction (`fixture_contacts.py`:11–28, `tlmpm.py` `_contact_impulse` :113–123). There is no grid boundary
  condition.
- The board impulse is not part of `bed.reaction`, so no runtime output reports board forces.
- The board's x/y size does not change the TLMPM grid (the domain comes from the particles: 5850 nodes in every row).

So the only guards against a blade inside the board are:
- planning: `goal_planner._relief_end` ends a solid-board cut on the board top;
- the action workspace floor at the board top;
- the preflight blade–board rule (distance ≥ 0 in contact phases);
- PhysX in Native. In §6 the commanded edge went 2 mm below the board top and the knife stayed on it.

**Recommendation: Option A with B (A+B).**
- **Option A.** End the armed descent with the edge's lowest point at the board top (h = 0). This is what the
  compiled flat-board sever plans already do; the reference edge never goes below local z = 0.
- **Option B.** The knife meets the board through real PhysX contact. Native measured 0.06 µm of penetration and a
  3–5 N reaction with the edge commanded 2 mm into the board.
- **Clean-cut window.** The kinematic window is h ∈ [0, +0.375] mm. A live controller that leaves the edge ≥ 0.5 mm
  above the board leaves a layer-0 bridge, which must be reported as an Option-B limitation, never hidden. WP-11 C5
  saw PRE-CUT oscillations of about ±0.2 mm.
- **Option C** (sacrificial or compliant board) needs a new deformable board model, which is a physics change. It is
  a documented study item only; the h = −0.25 mm diagnostic shows no cutting benefit.
- **Option D** (explicit slotted board) stays opt-in for cuts that must pass below the workpiece bottom (oblique or
  tilted planes, overtravel plans). It is never a hidden default.

## 6. Native checks

**Command:**

```
.venv/bin/python scripts/run_logged.py artifacts/wp12/logs/workcell_004_native.log -- \
  .deps/isaacsim-venv/bin/python scripts/run_wp12_flat_board_native.py --out artifacts/wp12/workcell_004 --gpu 0
```

**Run conditions:**
- Isaac Sim 6.2 Native, cuda:0, seed 42, 794 s wall (cuda:0 was shared with the kinematic study).
- Task `CFG-bar_flat_board-sever_mid-74cbdae6`.
- `ruler_knife_v2_y`, physical two-finger grasp (pregrasp PASS_STABLE_GRASP), HOME reset.
- The runner is unchanged except the opt-in flat-board table/appearance authoring.
- Kit exits through `app.close()`, so a return code of 0 is not task success. The status is in `result.json`.

| Check | Result | Verdict |
| --- | --- | --- |
| Start render: global D455, both wrist D405 and presentation overview/close-up | 5 PNGs per stage (start, after settle, knife over board) | PASS |
| Gravity settle: 2.0 s coupled, knife at HOME, cutting disabled | Horizontal centroid drift 4.2e-8 m; vertical settle −0.112 mm (elastic compression); max particle displacement 0.146 mm; 0 particles below the board top (lowest particle centre 0.43 mm above it); max speed 3.4 µm/s at the end; min det F 0.9977; no labels written | PASS |
| Knife cannot enter the board: rigid-only descent at 2 mm/s at local x = +0.12 m (beside the bar), commanded edge down to 2 mm **below** the board top, 1 s hold, lift | Knife lowest point (blade hull on the measured tool body pose) − board top: min −5.7e-8 m. The board reaction on the tool, inferred, averages (−0.43, −0.02, 3.03) N over the hold, max normal 3.45 N | PASS |
| Grasp slip during the press | Max 1.6 µm, 0.085 mrad; both fingers loaded throughout | PASS |
| Render after rigid-only controls | `capture()` fails the runner's wrist-camera extrinsic check, because extrinsics are refreshed only in coupled steps | NOT_RUN (runner limitation) |

**How the board force was measured.**
- The native tool contact sensor reports 1098 tool–board pair entries, but with zero impulse.
- Its tensor view is filtered to the fingers and palm only, so the "fixture tensor force" reads 0 by construction.
- The board reaction is therefore inferred quasi-statically: F_board = −(F_fingers,hold − F_fingers,free hover).
- Check: the free-hover finger force is 0.870 N, and the tool weight is 0.869 N.

**First run `workcell_003` (kept, FAIL).**
- The final render after the rigid-only probe failed the extrinsic check.
- Its measurements before that are valid. It used a rigid-only IK transfer from HOME:
  - penetration 8.3e-8 m;
  - inferred hold reaction (−0.99, 0.16, 4.97) N;
  - slip 23.8 µm / 1.62 mrad.
- The larger slip comes with the larger press force. In both runs the knife lever is the full 120 mm blade.

**The coupled approach did not fully converge.** The approach to the probe pose (runner Cartesian controller, cutting
disabled, 400 steps) ended 2.58 mm / 15.5 mrad from its target. The rigid descent started from there, so first
contact happened at commanded edge z = −0.76 mm. This is a controller tracking residual, reported as is.

**Images** (copied, ≤ 640 px):

| Before (wp11_workcell_v1, risers, +100 mm) | After (blade_master_flat_board_v1) |
| --- | --- |
| ![wp11 Test B start](../media/wp11/scenes/test_b_s1502_start.png) | ![overview](../media/wp12/flat_board_start_presentation_overview_rgb.png) |
| ![wp11 TRIM start](../media/wp11/scenes/trim_rho10_clear_start.png) | ![close-up](../media/wp12/flat_board_after_settle_presentation_closeup_rgb.png) |
| | ![global D455 start](../media/wp12/flat_board_start_global_d455_rgb.png) |
| | ![global D455, knife over board](../media/wp12/flat_board_knife_over_board_global_d455_rgb.png) |

**Compared with b.png:**
- Table, board, workpiece and the parked second Franka match.
- The knife looks different. In b.png a knife hangs vertically below the hand. With the end-held `ruler_knife_v2_y`, a
  vertical cut puts the edge line horizontal along +y and the hand beside the blade at −y. That hand pose is exactly
  why the table's front edge matters.
- The board colour is a light wood PreviewSurface, not a bamboo texture.

## 7. CPU tests

`tests/test_wp12_flat_board_workcell.py` covers:

- board bottom = table top; workpiece bottom = board top; particles neither below the board top nor floating;
- no riser or pedestal box in any layer;
- fixture boxes = board record in world;
- the Native authoring plan (visual/collider) and the PhysX/TLMPM friction come from the same numbers;
- the runner and the C2 model read the declared table, and the MuJoCo `static_table` and `static_flat_board` geoms
  have the record's sizes and positions (needs the Franka asset cache);
- a thickness or position change moves the contact plane, the deepest IK target and the preflight together, and a
  stale workpiece z is refused;
- off-board, yawed, table-top-mismatched, overhanging, implicitly slotted, through-slotted and negative-threshold
  configs are refused;
- the default board is solid and the slotted variant must be explicit, with grooves at the four slice planes;
- the action-workspace floor at the board top clamps a commanded edge below it;
- HOME is at local (0, 0, 0.188) with the hand pointing down;
- CUT, TRIM ρ = 10°, Test B (solid and slotted) and Test C compile, with payload round-trips;
- the wp11 compiled outputs are pinned (`tests/fixtures/wp12_wp11_compile_pin.json`, computed at `1c91605` before
  this change).

## 8. Failures and open items

**Failures (kept).**
- **`workcell_001` layout preflight: Test C FAIL** (blade vs bump −7.95 mm at `PRE_CUT:p1`). This is the C2
  single-hull artefact. The rerun `workcell_002` uses the staged per-protrusion hulls and passes.
- **`workcell_003` Native: FAIL.**
  - The end-of-run `capture()` after rigid-only controls failed the runner's own wrist-camera extrinsic check.
    `LiveIsaacSimCutting` refreshes wrist-camera extrinsics only in coupled `step()` calls.
  - Its measurements before that point are valid: penetration 8.3e-8 m, inferred hold reaction 5.0 N, slip 23.8 µm /
    1.62 mrad.
  - `workcell_004` renders after a coupled approach instead and skips the post-probe render (NOT_RUN).
- **Kinematic rows that fail by design:** S_P100 and S_P050 (h ≥ 0.5 mm leaves a bridge) and D40_000 (40 mm gap
  without overtravel).

**Runner and measurement limitations (reported, not fixed here).**
- **No render after rigid-only controls.** `rigid_probe_step` does not refresh camera extrinsics, so `capture()`
  afterwards raises. A fix would be a runner change (refresh extrinsics in `rigid_probe_step`), left for an opt-in
  follow-up.
- **The board-force sensor filter.** The native tool tensor contact view is filtered to the fingers and palm. The
  contact sensor lists tool–board pairs but with zero impulse. The board force is therefore inferred quasi-statically
  from the finger forces; the tool weight checks the inference to 0.6 mN. A direct reading needs the fixture/table
  paths in the tensor filters, which would be an opt-in change in `physical_contacts.py`.
- **TLMPM board forces are not reported at runtime.** The fixture impulse is not in `bed.reaction`. The kinematic
  study used a material momentum balance.
- **Controller tracking residual.** The coupled approach (runner Cartesian controller, cutting disabled) ended
  2.58 mm / 15.5 mrad from its target after 400 steps (581 s). The live through-cut on the flat board therefore has to
  show that the edge really reaches the board top within the clean-cut window (h ≤ 0.375 mm).
- **TLMPM has no blade–board contact** (§5). The guards are planning, the action workspace, the preflight and PhysX.
  An external controller that drives joints directly (as the probe does) relies on PhysX alone.

**Open items (next steps).**
- A live Native through-cut (sever_mid) on the flat board, with the measured edge–board distance during the cut.
  TRIM ρ = 10°, Test B and Test C live; return-home on the new HOME; friction/sliding; partial cut. Old-versus-new
  comparison of the spec §2.6 metrics (penetration, contact forces, displacement, success, bridges, grasp, IK and
  preflight failures, performance). Only the geometric preflight and the static/contact checks are done here.
- Near-horizontal skims depend on the table's front edge. A yawed workpiece or one far from the front edge can fail
  the preflight, and Phase 5 must count that as infeasible.
- The workpiece yaw must equal the board yaw: an oriented TLMPM fixture box does not exist.
- Wrist-camera framing on the lower workpiece has not been audited; the pictures exist but no visibility metric was
  computed. The board colour is a flat PreviewSurface; there is no bamboo texture.
- The compiled task id hashes the profile name and overrides, not the resolved record (same practice as
  `from_scenario`). Record changes need a new record version.

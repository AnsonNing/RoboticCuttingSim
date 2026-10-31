# ADR 0017 — `blade_master_flat_board_v1`: one cutting board lying on the table

Date: 2026-10-30. Status: accepted for implementation (WP-12 Phase 2, first part). Report:
[`WP12_WORKCELL.md`](../reports/WP12_WORKCELL.md). Evidence:
[`phase2_workcell_001.json`](../evidence/wp12/phase2_workcell_001.json),
[`phase2_throughcut_001.json`](../evidence/wp12/phase2_throughcut_001.json).

## Context

The WP-12 specification (Phase 2) asks for a BladeMaster-like scene, as in `/home/ningan/b.png`: a normal table, ONE
whole wooden cutting board lying directly on it, the workpiece directly on the board, the Franka holding a long
straight knife, the second Franka parked. There should be no risers or pedestals.

`wp11_workcell_v1` (ADR 0013, C2 decision) does something else:

- every support segment becomes a wood board plus a grey riser down to the table;
- the workpiece is raised +100 mm.

C2 raised the workpiece because of the hand, not the reach. With the end-held `ruler_knife_v2_y` the edge line is
parallel to the approach axis, so a near-horizontal skim (ρ = 10°) turns the Franka hand's finger-closing axis
vertical. The hand housing is 200 mm long along that axis, so it hangs about 80 mm below the trim plane. At +50 mm
the hand capsule went 16.5–17.3 mm into the table, so +100 mm was chosen. The layout study in this ADR measures the
same effect on the flat board (Franka 0 lowest points, relative to the board top, during TRIM ρ = 10°):

| Part | Lowest point | World y range |
| --- | --- | --- |
| Hand capsule | −80.4 mm | −0.204 to −0.124 m |
| Link 6/7 | −64.3 mm | −0.249 to −0.194 m |
| Wrist D405 | −30.4 mm | −0.197 to −0.174 m |
| Finger hull | −13.7 mm | −0.135 to −0.081 m |
| Pad | −2.4 mm | −0.098 to −0.082 m |

Everything that goes below the board top is at world y < −0.08 m, that is, on the grip side of the knife.

## Decision

1. **A new, versioned, opt-in workcell record.** `configs/wp12_workcells_v1.json` holds
   `blade_master_flat_board_v1` (fixture mode `flat_board_v1`), and `src/robotic_cutting_sim/flat_board_workcell.py`
   is pure numpy. A scene opts in with
   `"supports": {"kind": "flat_board", "workcell": {"profile": "blade_master_flat_board_v1", "overrides": {...}}}`.
   `overrides` may change only `table` and `cutting_board`. The record is not a `task_scenarios` scenario, and
   `wp11_workcell_v1` and every v1–v3 scenario are unchanged. Compiled wp11 tasks are pinned by
   `tests/fixtures/wp12_wp11_compile_pin.json`, computed at `1c91605` before this change.
2. **Geometry (defaults).**
   - **Table.** The top is at z = −0.01 m, the Franka mounting plane, so robot bases, the D455 and every robot-side
     constant stay where they were. The table spans x ∈ [−0.9, 0.9] and y ∈ [−0.075, 0.6] m and is 50 mm thick.
   - **Board.** One continuous solid board, 360 × 190 × 20 mm, centred at world (0, 0.03), yaw 0. Its bottom is on
     the table top and its top is at z = 0.01 m. Appearance: `bamboo_wood_v1` (wood colour, PreviewSurface; visual
     only).
   - **Workpiece.** Its bottom is on the board top (world z = 0.01 m), and validation refuses any other z
     (`workpiece_not_on_board`). Its footprint must lie on the board with a 5 mm margin. Its yaw must equal the
     board yaw, because no oriented TLMPM fixture box exists (otherwise an explicit error).
   - **Front edges.** The board's front (−y) edge is at y = −0.065 m and the table's at −0.075 m, 50 mm in front of
     the 30 mm wide bar. During a near-horizontal skim the hand and the wrist camera therefore hang past the table
     edge, as a person's knuckles hang over a counter edge. With the historical full table (front edge −0.6 m) the
     same TRIM fails: hand −56.5 mm, wrist D405 −10.4 mm into the table. CUT and Test B then have only 3.8 mm and
     −0.2 mm of link-6 clearance (5 mm required).
   - **Slotted board (option D).** Available only when declared (`cutting_board.kind = "slotted"`). Its slots are
     through-width grooves at board-frame x positions, `slot_depth_m` deep, which is less than the thickness, so the
     board stays one continuous body: a base slab plus top segments. A slot without `kind: "slotted"` is an error.
3. **One source for every consumer.** The record generates:
   - the board `MaterialFixtureBox` es. They are the TLMPM fixture boxes and the Native cubes; the visual box is the
     PhysX collider box. They are also the planner's support boxes.
   - the table box: the Native `/World/Table` translate and scale, and the C2/`motion_preflight` `static_table`.
   - the support plane: workpiece-local z = 0 = board top = table top + thickness.
   - HOME and the action workspace.

   No riser or pedestal exists in any layer. The C2 checker reads `task.table_box_world` only when a task declares
   it; every other task keeps the historical table.
4. **HOME.** The knife edge is at the same workpiece-local pose as in wp11 (0, 0, 0.188) m, with the hand pointing
   straight down and the blade hanging (a.png). That is world z = 0.198 m. q_home comes from the runtime Newton IK,
   seeded with the wp11 q_home:
   [−0.08763, −0.26645, −0.17753, −2.43878, −0.05656, 2.17558, 0.55855] rad
   (position error 0.97 µm, smallest joint margin 0.633 rad).
5. **No silent board penetration.**
   - The action workspace z minimum is derived from the board: 0 (the board top) for a solid board and −slot depth
     for a slotted one. A commanded edge centre below a solid board top is CLAMPED and reported.
   - In the collision preflight, in contact_free and contact_allowed samples the blade may reach the board top
     (signed distance ≥ `preflight.blade_board_contact_min_distance_m` = 0; negative values are refused) but never
     enter it. Free-space samples keep the C2 1 mm clearance.
   - The planner's existing support rule already stops a through-cut on a solid board with the edge exactly on the
     board top (h = 0).
   - In Native, the knife meets the board only through PhysX contact (measured below).
6. **Contact architecture.** Material and rigid contact are not counted twice:
   - material ↔ board: TLMPM fixture contact only, with the task material friction;
   - knife/robot ↔ board and table: PhysX only, with the record friction (`task_material` makes both equal);
   - knife ↔ material: the TLMPM blade coupling.

   The TLMPM solver has no knowledge of the board as an obstacle for the kinematic blade. See the through-cut study
   for what this means.
7. **Through-cut on the solid board (spec §2.4).** Option A with B; see "Through-cut decision" below.

## Through-cut decision (spec §2.4)

**Data.** Kinematic, `tlmpm_v5`, law 8, 1 mm particles, 4 mm grid, 0.1 ms, seed 42, CUT-01-R bar, `ruler_knife_v2_y`
([evidence](../evidence/wp12/phase2_throughcut_001.json)):

- **Solid board, clean range.** h = 0, +0.25 and +0.375 mm give two persistent pieces, all 30 layer-0 cross-plane
  links conflicting and 0 bridges.
- **Solid board, bridged range.** h = +0.5 mm leaves 16 layer-0 bridges; +1.0 mm leaves 172.
- **Penetration diagnostic.** h = −0.25 mm (the edge inside the board; diagnostic only, never adoptable) is identical
  to h = 0.
- **Slotted variants.** The 8 mm slot separates at −4 mm and at 0. The CUT-01-R 40 mm gap needs the −4 mm
  overtravel: at 0 the bar sags into the gap and 48 bridges remain.
- **Material–board contact.** No particle centre enters the board. The material–board force equals the weight.

TLMPM has no blade–board interaction: the kinematic blade acts on material only, and fixture boxes act on material
only, through a per-particle penalty.

**Native.** With the edge commanded 2 mm below the board top, PhysX held the knife on the board: 0.06 µm penetration,
3.0 N inferred hold reaction (5.0 N in the first run), grasp slip ≤ 1.6 µm.

**Decision: Option A with B.**
- The armed descent of a through-cut on a solid board ends with the edge's lowest point at the board top (h = 0). This
  is the existing planner support rule; the compiled flat-board plans never go below local z = 0.
- The knife meets the board only through real PhysX contact. That contact is guarded by the action-workspace floor
  and the preflight rule.
- The kinematic clean-cut window is h ∈ [0, +0.375] mm for this material and resolution. A live edge that stays
  ≥ 0.5 mm above the board leaves a layer-0 bridge, and that must be reported as an Option-B limitation, never
  hidden.
- Option D (slotted) stays explicit and opt-in, for cuts that must pass below the workpiece bottom.
- Option C (sacrificial or compliant board) needs a new deformable-board physics model. It is a documented study item
  only, and the penetration diagnostic shows no cutting benefit.
- The edge never passes through the rigid board, silently or otherwise.

## Consequences

- WP-11 scenes, ids, evidence and the +100 mm workcell are unchanged and still selectable.
- The flat board is the recommended default for new WP-12 scenes. Near-horizontal skims with the end-held knife
  depend on the table's front edge: a workpiece far from the front edge, or a yaw that puts the grip over the table,
  fails the preflight. The preflight reports this as SCENE_INFEASIBLE, never as a silent collision.
- The compiled task id hashes the scene (profile name and overrides), not the resolved record. This is the same
  practice as `wp11_workcell_v1` via `from_scenario`. The record is versioned and frozen per version, and every
  receipt carries the resolved geometry (`ConfiguredTaskSpec.to_dict()["workcell_geometry"]`).
- Isaac Lab, `presentation_camera` and `robot.py` keep their hard-coded robot bases. The flat board does not move
  them.

# ADR 0013 — `ruler_knife_v2` (end-held blade) and near-horizontal TRIM orientation metrics

Date: 2026-10-30. Status: accepted for implementation. The blade/handle
arrangement is chosen by the C2 preflight, as recorded below.

## Context

The user's reference image (`/home/ningan/a.png`) shows a Franka hand whose two
fingers pinch a slim grip at **one end** of a long, flat, straight-edged metal
blade. The blade extends away from the hand along its length, and its
straight cutting edge runs along that length. The user asked explicitly that
the knife look like this. WP-11 (Phase B/C) also requires near-horizontal
0–10° skims.

`ruler_knife_v1` is `tool_profile.py`:
- Blade: 120 × 40 × 1.5 mm, 25° double bevel.
- Handle: a 100 × 22 × 40 mm box between the fingers, centred on the blade
  **spine**.
- Mount: the blade hangs 40 mm below the handle along hand +z, with the edge
  line along hand x.

Measured differences from the reference:
1. **Grip location.** v1 is gripped along the spine centre; the reference is
   gripped at one end. In v1 the long edge is parallel to the palm, whereas
   the reference blade extends out of the hand.
2. **Horizontal skims are impossible with v1.**
   - With the blade plane near horizontal, the handle sits 40–80 mm behind the
     edge and ±11 mm about the blade plane.
   - It therefore follows the edge across the workpiece top, 11 mm below the
     trim plane.
   - The finger pads cross the trim plane by about 1.8 mm already at 25°
     (WP-10C), which is why the final TRIM uses a 35° rake.

The blade itself (proportions, thickness, bevel) already matches the
reference and stays unchanged. Only the grip changes.

## Decision

Add `ruler_knife_v2` as a new, versioned profile. `ruler_knife_v1`, its tasks
and the `final_v4` demos stay unchanged and reproducible.

### Geometry

- **Blade:** identical to v1 (edge length 120 mm, depth 40 mm, thickness
  1.5 mm, 25° double bevel, steel 7800 kg/m³). The TLMPM cross-section
  polygon is therefore identical to v1's.
- **Handle:** a slim grip (default 70 mm long along hand +z, 22 mm across the
  finger closing axis as in v1, 24 mm in the third axis, 1100 kg/m³). It is
  centred at the finger pads, `HAND_FROM_TOOL` as in v1. The blade starts at
  the handle's far face along hand +z and extends a further 120 mm along
  hand +z, so the **edge line is parallel to the approach axis**. The blade
  and handle meet at that face with disjoint interiors.
  *Implementation amendment (Phase B, 2026-10-30):* a 70 mm grip centred at
  the pads reaches the actual Native `panda_hand` palm collider. The palm face
  is measured at hand z = 0.0700 m (`artifacts/wp11/phaseB/view_v2_y_001`,
  pregrasp rejected for palm contact). A centred grip can be at most 67.3 mm,
  so the registered profiles use **60 mm**, which leaves 3.65 mm of clearance.
- **Arrangement parameter `blade_normal_hand_axis`** (the C2 preflight
  selects one):
  - `"y"` (A): the blade normal is the finger closing axis, as in v1. The
    fingers squeeze across the blade thickness direction. The blade width
    spans hand +x from the spine, which is flush with the handle face, to
    the edge.
  - `"x"` (B): the blade normal is hand x. The blade width spans the finger
    closing axis.
- **Single source of truth.** One `ToolProfile` generates the visual mesh,
  the PhysX colliders, the mass/inertia/COM, the grasp frame, the edge frame
  (`hand_from_edge`), the TLMPM blade polygon and the blunt handle obstacle
  box, exactly as for v1. `require_profile_consistency` must hold.
- **Grasp.** Physical two-finger friction grasp only: the same
  `ToolGraspConfig` limits (≤ 20 N per finger, μ = 0.6) and the same
  settle/monitor gates. There is no fixed joint, hidden constraint or pose
  sync.

### Orientation metrics for skims (used by the preflight, controller and reports)

| Symbol | Meaning |
| --- | --- |
| \(n_t\) | Unit normal of the target surface (world +z for a flat target) |
| \(c\) | Unit cutting direction in the target plane |
| \(e\) | Unit edge-line direction |
| \(n_b\) | Unit blade-plane normal |
| \(m\) | Unit direction of the measured edge velocity |

These quantities define three different constraints:

| Metric | Definition | Target |
| --- | --- | --- |
| Edge tilt \(\tau\) | \(\arcsin\lvert e\cdot n_t\rvert\) (the edge line is parallel to the surface) | 0 |
| Edge skew \(\psi\) | Angle between \(e\) and \(n_t\times c\) (the edge is square to the cut) | Declared per task; 0 by default |
| Blade rake \(\rho\) | Signed angle between \(n_b\) and \(n_t\), about \(e\); positive when the blade face rises behind the edge | 0–10° preferred, 35° baseline |
| Motion elevation \(\mu\) | \(\arcsin(m\cdot n_t)\) (the motion is parallel to the surface) | 0 during the skim |

A pose can have \(\rho = 0\) and still tilt its edge (\(\tau \ne 0\)), or move
non-horizontally (\(\mu \ne 0\)). Every TRIM receipt reports all four metrics.

## Consequences

- Each new task, scenario and demo set that uses v2 is versioned. The v1/v2
  task thresholds and `final_v4` are untouched.
- If neither arrangement can skim near-horizontally on the existing workcell,
  a versioned raised scenario may be added, for example more board/table
  clearance. Whatever is needed is recorded.
- Cutting-law compatibility at 0–10° (law 8's 15° plowing gate) is evaluated
  separately at the material level (WP-11 Phase C4) before any live run.

## Decision record — C2 preflight (2026-10-30, `c2_preflight_003`, [evidence](../evidence/wp11/phaseC2_preflight_001.json))

The preflight used Newton IK with the runtime's `DualFrankaController`, and
MuJoCo distances on the official Franka colliders plus finger hulls, the v2
handle and blade, the v5 wrist D405s, the D455, board, riser, table,
workpiece and the parked Franka 1. It covered 84 TRIM candidates and 30
follow-ups (CUT, 20° oblique, SLICE-PUSH).

- **Below 12.5° the bevel blocks the skim.** At ρ < 12.5° the double bevel
  puts the blade's lowest point below the edge. With the edge at target +
  0.2 mm, that point sits 0.55 mm (0°) or 0.25 mm (5°) inside the kept
  substrate; at 10° it clears by 0.05 mm. Every robot, tool and workcell
  check passes at 0° and 5° wherever it passes at 10°, so the binding
  constraint is the blade geometry, not the robot.
- **Workcell choice for all tasks:** `ruler_knife_v2_y` with the tip toward
  world +y, the workpiece raised **+100 mm**, and TRIM at **ρ = 10°**. TRIM,
  vertical CUT, the 20° oblique cut and SLICE-PUSH all pass from one q_home,
  `[−0.0875, −0.3457, −0.1643, −2.2755, −0.0593, 1.9335, 0.5643]` rad, with
  fingers at 0.0112 m each.
  - Tightest margins: the blade passes 3.0 mm above the board during the
    push, as designed; the handle clears by 15.5 mm; the worst joint margin
    is 0.77 rad.
  - HOME has the hand pointing straight down with the blade hanging below
    the fingers, as in `a.png`.
- **The +50 mm placement (`blademaster_workcell_v1`) cannot serve all
  tasks.** There only `v2_x` with the tip toward −y passes TRIM at 10°, and
  that same choice fails CUT, oblique and SLICE-PUSH: the hand collider
  sits 15.3 mm into the table.
- **Chosen:** the all-task workcell becomes the versioned scenario
  `wp11_workcell_v1` (+100 mm). The near-horizontal TRIM target is
  ρ = 10°. A 0–5° skim with a flat-face-down chisel variant stays possible
  only if Phase C4 shows separation at the material level and a chisel
  preflight passes. Otherwise ρ < 10° is reported as blocked by the bevel
  geometry.

## Decision record — C4 material level (2026-10-30, [evidence](../evidence/wp11/phaseC4_trim_material_001.json))

Settings: law 8, 1 mm particles, 0.1 ms, kinematic, seed 42, edge at
target + 0.2 mm.

**Double bevel (`ruler_knife_v2_y`)**
- ρ = 5°, 10°, 12.5°, 15°, 25° and 35° all separate the dome: 93–98%
  removed, substrate share ≥ 0.99996, zero code-0 bridges.
- ρ = 0° fails: 70% removed, or 87% with exact clearance. The lower bevel
  face sits 0.55 mm below the target plane and depresses the dome base, so
  its bottom layer is labelled with the substrate.

**Chisel (`ruler_knife_v2_y_chisel`, flat face down)**
- Passes at 0°, is marginal at 5°, and fails at 10°.
- The cause is law 8's plowing gate. It measures motion against the blade
  plane (5–10°), but the chisel's top face pushes the chip at ρ + 25°.
  Ejected chip particles therefore stay code 0 and bridge.
- This is a law-8 boundary finding for Phase F. It is not addressed with a
  new law version.

**Final TRIM configuration:** `ruler_knife_v2_y`, double bevel, ρ = 10°,
edge at target + 0.2 mm, in `wp11_workcell_v1`. This meets the 0–10°
near-horizontal target and is consistent with the C2 preflight. A true 0° is
reported as blocked for the double bevel by its bevel geometry. The chisel
0° option stays a documented material-level PASS that is not pursued live,
because it would need its own CUT, oblique and SLICE validation.

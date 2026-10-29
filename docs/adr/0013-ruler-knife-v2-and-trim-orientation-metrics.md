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

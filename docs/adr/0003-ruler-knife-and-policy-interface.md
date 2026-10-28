# ADR 0003 — Ruler knife profile and policy-facing control interface

Date: 2026-10-28. Status: accepted for WP-10 part A (CPU only; GPU NOT_RUN).

## Context

The user wants Franka 0 to physically hold a ruler-like straight knife and use
it to complete cutting tasks (severing, removing bumps) with any motion a
policy, RL agent or RSI agent chooses, with state and image observations.
The retained tool is a 50 mm wide, 35 mm (CUT) / 12 mm (TRIM) deep wedge on a
short shaft; the action API raised on unreachable targets, had no per-step
bounds and a zero reward. The user delegated the tool dimensions and goals.

## Decisions

1. **ruler_knife_v1**: 120 mm edge, 40 mm blade depth, 1.5 mm steel, symmetric
   double bevel with a 25 deg included angle (bevel 3.4 mm). Rationale: longer
   than the 30 mm (and future 60 mm) workpiece width for oblique/yawed cuts;
   depth above the 20–25 mm workpiece height so the spine clears the material
   after a through cut; double bevel to avoid one-sided steering in straight cuts.
2. **Spine-mounted handle** (hand-frame half extents 50 x 11 x 20 mm, 1100 kg/m3),
   edge center on the grasp axis. Cutting force then passes through the finger
   pads with no lever arm across the grasp; a kitchen-knife layout (handle
   collinear with the edge) would put 100 mm of lever arm on pad friction
   torsion. The 22 mm width matches the existing grasp settings
   (initial 0.014 m, grip 0.0112 m). Total mass about 150 g.
   Consequence: flat skimming (TRIM) needs a rake tilt so the handle clears the
   trimmed surface; that is an action the policy can choose.
3. **One profile, every consumer**: colliders/mass, visual mesh, TLMPM blade
   polygon and hand-to-edge mount derive from `ToolProfile`; consistency is
   checked against the profile instead of the legacy wedge dimensions.
4. **Frozen v1 solver untouched**: `tlmpm_v2.py` is a copy with the `new`
   argument renamed (Warp CPU compile) and a convex-polygon blade SDF. Its
   default legacy triangle matches a renamed v1 bit for bit on CPU. v1 stays
   the default; v2 is selected only with a tool profile.
5. **Opt-in policy interface**: delta actions, per-step limits (2 mm, 2 deg per
   20 ms), workspace box, IK rejection with hold, state profile and reward are
   all opt-in; defaults reproduce the historical API.

## Not decided here

Cutting law v2 (energy-gated separation, slicing, blunt non-edge contact,
multi-cut), runtime integration, materials and RL speed: see
`docs/WP10_SPECIFICATION.md` (WP-10B to WP-10G).

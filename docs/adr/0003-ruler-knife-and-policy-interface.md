# ADR 0003 — Ruler knife, general knife motions and policy-facing control

Date: 2026-10-28. Status: accepted for WP-10 parts A and B (CPU-verified;
GPU and live runtimes NOT_RUN).

## Context

The user wants Franka 0 to physically hold a ruler-like straight knife and use
it to complete cutting tasks (severing, removing bumps) with any knife motion a
policy, RL agent or agentic-control/LLM agent chooses, with state and image observations,
and later actual videos of those controllers. The retained tool is a 50 mm
wide, 35 mm (CUT) / 12 mm (TRIM) deep wedge on a short shaft; the action API
raised on unreachable targets, had no per-step bounds and a zero reward. The
user delegated the tool dimensions and secondary goals.

## Decisions

1. **ruler_knife_v1**: 120 mm edge, 40 mm blade depth, 1.5 mm steel, symmetric
   double bevel with a 25 deg included angle (bevel 3.4 mm). Longer than the
   30 and 60 mm workpiece widths for oblique and yawed cuts; deeper than the
   20–25 mm workpiece heights so the spine clears the material after a through
   cut; double bevel to avoid one-sided steering in straight cuts.
2. **Spine-mounted handle** (hand-frame half extents 50 x 11 x 20 mm,
   1100 kg/m3) with the edge center on the grasp axis, so cutting force passes
   through the finger pads with no lever arm across the grasp (a kitchen-knife
   layout would put 100 mm of lever on pad friction torsion). The 22 mm width
   matches the existing grasp settings. Total mass about 150 g.
3. **One profile, every consumer**: colliders, mass, visual mesh, TLMPM blade
   polygon and the hand-to-edge mount derive from `ToolProfile`; consistency is
   checked against the profile instead of the legacy wedge dimensions.
4. **Frozen v1 solver untouched**: `tlmpm_v2.py` renames the C++-keyword
   argument (Warp CPU compiles) and generalizes the blade to a convex polygon;
   its default (law 1, legacy triangle) matches a renamed v1 bit for bit on CPU.
5. **Cutting law 2** (opt-in; default for v2 tasks): cutting work = push work +
   drag work along the edge, times cos^2 of the angle between the edge's relative
   motion and the blade plane; separation labels only where the edge has cut;
   blunt obstacle boxes (handle) push without cutting. Rationale: slicing and
   sawing must cut (v1 ignores them), and armed-but-not-cutting motions must not
   leave labels that later cuts inherit. The earlier claim that v1 wrongly
   severs a 15 deg misaligned push was withdrawn: that is a legitimate oblique
   cut, and v1 severs nothing under true sideways pushes or flat presses.
6. **cutting_tasks_v2**: `CUT-01-R`, `TRIM-01-R`, `CUT-02` inherit their base
   task's v1 thresholds unchanged; only tool, geometry, fixtures and path differ.
   v1 task classes, receipts and protocol hashes are unchanged.
7. **TRIM-01-R at a 25 deg rake** about the edge line: with a spine handle,
   flat skimming would drive the handle and finger pads into the trimmed
   surface; at 25 deg the edge is the lowest tool point and the handle clears
   the plane by about 7 mm (finger clearance estimated, to verify in WP-10C).
8. **Opt-in policy interface**: delta actions (frame selectable per action),
   per-step limits (2 mm, 2 deg per 20 ms), workspace box, IK rejection with
   hold, state profile, `cut_index` passed through to the material, reward and
   Gym adapter. Defaults reproduce the historical API.
9. **Reward on severed bonds** (`cutting_reward_v1`): progress counts target-
   plane columns with a reference bond whose two particles carry opposite cut
   sides. A label-count reward (v0, never used in a recorded run) could be
   gamed by one-sided labels.
10. **Kinematic backend** for controller validation, RL pre-training and
    cutting-law probes. Results are always labelled
    `kinematic_ideal_tracking` and never substitute live-runtime evidence.

## Consequences and open items

GPU and live-runtime validation, runtime integration of the profile tool,
finger-pad obstacles, materials, multi-cut evaluation, RL speed and the
end-to-end controller validation with actual robot videos are specified in
`docs/WP10_SPECIFICATION.md` (WP-10B to WP-10G).

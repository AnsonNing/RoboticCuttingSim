# ADR 0005 — explicit WP-10 scene candidates and unvalidated law 3

Date: 2026-10-29. Accepted for versioning; scientific acceptance remains separate.

Actual USD finger pads at the original 25-degree TRIM rake cross the target
plane by about 1.8 mm. Alternate Newton IK branches also put the palm through
the original table by about 15.97 mm. Preserve those failures and all original
task definitions. Add optional scenario files: v1 raises the workpiece and
anchored supports by 50 mm; v2 also chooses a 35-degree TRIM rake. Material
geometry and scoring thresholds remain inherited. Only Franka 0 receives a
scenario-specific IK seed. Franka 1 keeps its normal park; camera rig v5 keeps
fixed hand-local mounts without aiming or object-visibility gates.

The v2 candidate's 401-pose Newton preflight has at least 8.27 mm pad-to-target
clearance and no penetrating robot/table/self contacts in that model. This is
geometry evidence only. No v2 live smoke or full task is claimed; fixture and
camera/robot intersections still require dedicated audits.

Add two provisional food profiles using stiffness orders from
[Hutchings et al., Dairy versus non-dairy cheese texture, Table 3](https://doi.org/10.1111/jtxs.12863):
reported cream-cheese and cheddar elastic moduli are 25,933 and 477,861 Pa.
Use rounded E = 26,000 and 480,000 Pa. The other parameters are explicitly
engineering assumptions. Select nominal dt = 100 and 20 µs using CFL <= 0.2
and record every dt-sweep result. These are not calibrated constitutive laws;
volumetric toughness is not interpreted as area fracture energy.

Even ideal-tracking law-2 TRIM at 1 mm retains a connected substrate/bump.
Keep `tlmpm.py`, `tlmpm_v2.py` and their historical records intact. Preserve a
new, explicitly selected `tlmpm_v3.py` draft to investigate swept-edge side
classification and one-sided closure of energized MPM support fields. Contact
work is still required and no particles are deleted. This support closure is
a process-zone hypothesis, not an energy-convergence result. The draft has
only passed syntax/CPU kernel compilation. Before use, require focused
classification tests, armed-withdrawal/side/flat/handle negative controls,
law-1/2 regression, and formal-resolution skimming evidence. Runtime presets
continue to select law 2; no training or task-success claim relies on law 3.

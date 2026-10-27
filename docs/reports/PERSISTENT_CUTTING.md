# Experimental persistent cutting

This independent implementation adds a **new experimental solver** alongside the
Newton implicit-MPM and CRESSim probes. It does not change their recorded results
or claim to be the BladeMaster authors' code. Scientific validation is ongoing.

The continuous heightfield is sampled once. Every material point starts with
zero cut history, fixed mass and a reference position. A moving finite blade
assigns persistent ternary side digits online; these determine separate logical
reference-grid states. Compatible states exchange momentum conservatively.
Opposite cut faces retain separate states after tool withdrawal and interact by
equal-opposite contact impulses rather than reconnecting. No particle is deleted,
duplicated, hidden or driven as a separate fragment.

`tlmpm.py` implements linear total-Lagrangian MPM, MUSL, Hencky J2 plasticity,
PIC/FLIP, finite rigid tray boxes, a surrounding support table, a triangular wedge,
and current-position particle contact. Material and blade parameters are
provisional. The host separately rejects nonfinite/inverted/exploding states;
constitutive logarithm safeguards do not count as acceptance.

The topology idea follows [BladeMaster](https://arxiv.org/abs/2609.27342), with
[CPIC](https://yuanming.taichi.graphics/publication/2018-mlsmpm/) as an earlier
discontinuity-aware MPM reference. The implementation uses public mathematical
descriptions; no paper code or text is copied. Source review of
[CRESSim-MPM](https://github.com/yafei-ou/CRESSim-MPM) and its actual withdrawal
microtest show why a current-tool crossing test alone does not retain a crack.
[DiSECt](https://github.com/NVlabs/DiSECt) prepares its cut interface before the
simulation. [STC-MPM](https://arxiv.org/abs/2610.06198) separates constitutive
weakening from kinematic discontinuity and does not provide a validated drop-in
persistent-cut solver for this environment.

## Reproduce

```bash
.venv/bin/python scripts/run_logged.py artifacts/my_cut.log -- \
  .venv/bin/python scripts/run_cutting_probe.py --case cut \
  --output-dir artifacts/my_cut --device cuda:0
.venv/bin/python scripts/run_cutting_probe.py --case cut_recontact \
  --output-dir artifacts/my_recontact --device cuda:0
```

Each run refuses to overwrite output and records a manifest, exact tool sequence,
particle IDs/masses, all sampled positions/velocities/side codes, final deformation
and plastic state, reaction traces and contact diagnostics. Mechanical-component
analysis includes shared reference stencils, nonzero stress gradients and
unassigned-field bridges. A visual gap or label count alone is insufficient.

## Early measurements, retained for provenance

On 2026-10-28, `tlmpm_cut_003` ran 19,340 particles for 2.5 simulated seconds:
one initial mechanical component became two, comprising a 1.68 g/140-point chip
and a 230.4 g/19,200-point substrate. The chip moved another 85.9 mm after
withdrawal and landed on the surrounding table. Maximum speed was 1.064 m/s;
minimum det(F) was 0.9195. No existing side digit changed.

`tlmpm_recontact_001` used a shorter sweep and vertical withdrawal: the chip fell
onto the substrate, producing 182,244 post-withdraw contact-substep pairs while
the mechanical components remained separate. Maximum sphere overlap was 0.809 mm
at this coarse 2 mm particle spacing. This is an explicitly reported contact
error, not a nonpenetration or material-calibration pass.

These early runs precede the final work-threshold accounting refinement. They
are intermediate evidence, not frozen acceptance or convergence results.
`tlmpm_cut_001/002` failed when falling chips met a wall modeled as an unbounded
half-space. Raw failures remain intact. The finite-box correction and a separate
MUSL position/velocity consistency correction are recorded in the implementation.

Pending: frozen same-seed repeats, timestep/grid controls, sequential cuts,
quantified recontact, material calibration, actual articulated coupling, and
surface/camera validation of these cutting runs.

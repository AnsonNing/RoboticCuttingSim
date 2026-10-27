# Validation boundaries

Evidence index: [validation_summary.json](evidence/validation_summary.json).
Per-work-package commands and numerical results are in `docs/reports/`.

| Test family | Current finding |
|---|---|
| CPU geometry/seed/trajectory/API |23 tests pass, including connected material, collider buffers and observation/capability boundaries |
| PHY-01 support |Newton2s finite supported settling; CRESSim candidate fails stability/support |
| PHY-02 indentation |Newton yield-dependent residual exceeds elastic control; uncalibrated, numerical residual remains |
| PHY-03 blade contact |Newton trace/video and pre/post-projection diagnostics; no robot coupling |
| PHY-04/05 physical cut and recontact |NOT_PROVEN; no persistent discontinuity or chip classification implemented |
| PHY-06 mass |Constant-particle global accounting only; no chip/substrate budget |
| PHY-07 convergence |4/3/2mm exploration; initial quadrature and dt differ, no convergence pass |
| PHY-08 reset |Fresh process/model/solver same-seed121frame equality on one GPU; in-place history reset NOT_RUN |
| VIS-01 ideal plane |Custom ideal ray camera numerical depth/mask PASS; D455 NOT_RUN |
| VIS-02 dynamic surface |Programmable arrays from recorded MPM deformation PASS; live Isaac sensor BLOCKED |
| VIS-03 true cut |NOT_RUN because physical cut not established |
| ROB-01 static scene |Official dual-Panda/dual-D455 composition and sensor optical-axis placement PASS; articulation/collision/controller NOT_RUN |
| ROB-02/03 robot motion/feedback |NOT_RUN; currently kinematic blade independent of arms |
| ENV-01/03 facade/observation |Live120-step MPM+RGB-D facade PASS; CPU contracts exclude force/oracle/global, direct pose controls available |
| ENV-02 variants |D0/D1 generation tested; D2 and formal benchmark randomization pending |

CLI PASS means the named engineering scope executed and its stated checks passed.
It never upgrades PHYSICS_PROVEN/VISION_PROVEN automatically. Failed CRESSim
experiments return nonzero. Missing official GA dependencies and CUDA13 runtime
are BLOCKED, not successful skips. All four final research-ready gates remain false.

The broad runtime guard (peak speed<2m/s and particle Z≥-1mm) is an engineering
failure screen for this centimetre-scale testbed, not a calibrated physical
acceptance threshold. Formal thresholds remain to be frozen after convergence
and material validation. No result claims physical real-plasticine equivalence.

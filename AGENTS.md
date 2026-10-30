# RoboticCuttingSim

Read `RoboticCuttingSim_Implementation_Plan.md` and `docs/PROGRESS.md` first.
Keep work inside this independent repository. Do not modify other projects,
upgrade system drivers, train policies, or push without an explicit user request.
Implement a small, executable physics testbed before integrating robot arms.
Never call geometric deletion, presegmented bumps, or renderer masks physical cutting.
Record exact commands, versions, seeds, failures, timing and artifact paths.
Report PASS / FAIL / BLOCKED / NOT_RUN independently; a skipped test is not a pass.
Material parameters are provisional and uncalibrated. SI units, world Z up,
quaternions XYZW. Keep force and oracle material state out of vision observations.
The user prioritized a usable research scene over strict stage gates: static
dual-arm workcell and the environment API may proceed independently of formal
cutting proof. Keep real supported capabilities distinct from unimplemented
persistent cutting, articulated control and RTX sensors; do not block routine
scene work on scientific acceptance labels.
Do not overwrite experiment output. Preserve unsuccessful experiments.
Run CPU tests with `.venv/bin/python -m pytest`; run GPU probes separately.
Current camera rig is v5: both wrist D405s have an identity optical rotation
relative to their actual hand, with the same fixed side mount. Franka 1 stays
at the shared normal safe park. Do not restore wrist look-at aiming,
observation-pose optimization or a secondary object-visibility gate. Preserve v1-v4
configurations/evidence. Evaluation v2 is a read-only consumer with separate
integrity/outcome/quality layers; `evaluate_oracle` and no-argument live
`evaluate()` retain the frozen v1 precision benchmark. Never modify v1
thresholds, physics or records to improve a score.

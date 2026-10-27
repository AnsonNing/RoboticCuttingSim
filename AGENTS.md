# RoboticCuttingSim

Read `RoboticCuttingSim_Implementation_Plan.md` and `docs/PROGRESS.md` first.
Keep work inside this independent repository. Do not modify RSI or other projects,
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

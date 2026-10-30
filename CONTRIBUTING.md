# Contributing

Read the implementation plan and current gate states in `docs/PROGRESS.md`.
Keep changes small and commit separately by independently reviewable work package.
Use Python 3.12 and the repository venv; CPU unit checks are
`.venv/bin/python -m pytest -q`. GPU physics and camera experiments are explicit CLI
commands documented in `docs/SETUP.md` and work-package reports; a CPU pass does
not certify GPU physics. Check live GPU usage before starting expensive work.

Every physics/backend change needs a fresh output directory, manifest, trace and
exit receipt. Preserve failed experiments. Do not commit binary dependencies,
videos, raw states or vendor assets. Add source/license references for dependencies.
Do not use particle deletion or presegmented material as physical cutting, broaden
scope into policy training, or touch other projects. Public dissemination does not waive asset
licenses or physics/vision acceptance gates.

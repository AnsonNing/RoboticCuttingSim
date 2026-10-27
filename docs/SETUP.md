# Setup

The primary handoff is a usable research workcell and environment API. After
installation, run the integrated demonstration:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_research_demo.sh artifacts/my_research_demo
```

This builds a locally composed dual-Panda/dual-D455 static USD, runs120 live MPM
steps with ideal RGB-D, and exports a combined measured-state replay. Vendor USD
dependencies (about21.4MB) download once into ignored artifacts. For an existing
complete official hierarchy use `scene --preset workcell --asset-root PATH`.
See `WORKCELL.md` and `ENVIRONMENT_API.md` for supported capabilities.

Python 3.12 is required. All installs are isolated in this repository. Do not
install packages into RSI or alter the machine driver during this procedure.

```bash
cd /home/ningan/RoboticCuttingSim
/home/ningan/.local/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/fetch_sources.py
.venv/bin/python -m pytest -q
```

`requirements.lock` describes the standalone CUDA 12 Newton probe, **not** a full
Isaac Lab GA installation. For GA use the pinned checkout's official uv workflow
after resolving the driver gate in [COMPATIBILITY](COMPATIBILITY.md). No old Isaac
examples are substituted. The current machine can reproduce the blocker:

```bash
.venv/bin/python -m robotic_cutting_sim.cli doctor --output-dir artifacts/environment/fresh_doctor
.venv/bin/python scripts/official_smokes.py artifacts/fresh_official_lab
```

Both return nonzero when the required stack is blocked. Raw command, stderr and
exit codes are preserved. A source-only attempt does not replace a full GA smoke.

Build the independent CRESSim library and process adapter (CUDA 12.8, RTX 5090):

```bash
bash scripts/build_cressim.sh
```

This uses four compiler jobs and sm_120 only. Adjust toolkit path and architecture
explicitly on a different host. CRESSim is not thread safe; run it in its own process.

Create a static scene, run an actual 2 s / 120-frame blade experiment, and replay
its extracted surface through programmable ideal cameras:

```bash
.venv/bin/python -m robotic_cutting_sim.cli scene --output-dir artifacts/scene_new
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg .venv/bin/python -m robotic_cutting_sim.cli physics-probe --case indent_unload --output-dir artifacts/indent_new
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg .venv/bin/python -m robotic_cutting_sim.cli physics-probe --case through_bump_cut --output-dir artifacts/scrape_new
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg .venv/bin/python -m robotic_cutting_sim.cli physics-probe --solver cressim --case through_bump_cut --substeps 100 --output-dir artifacts/cressim_new
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 .venv/bin/python -m robotic_cutting_sim.cli camera-check --input artifacts/indent_new --output-dir artifacts/rgbd_new
```

Check live GPU usage first. `CUDA_VISIBLE_DEVICES=1` exposes physical GPU 1 as
`cuda:0` to each process. Use fresh output paths; existing evidence is preserved.
Cold JIT compilation affects timing and is included in receipts. `scripts/run_logged.py`
can wrap any command and records wall time, exit code and sampled device-wide VRAM.
Run `--help` on implemented CLI commands. There is no train or RSI command.

CRESSim is an experimental candidate with known numerical failures; its command
currently returns FAIL/exit1 and preserves evidence. It is not required for using
the working Newton environment or building the static workcell.

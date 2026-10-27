# Setup

Use Python3.12 and an isolated environment inside this repository. The runnable
backend is standalone Warp/Newton/MuJoCo/OpenUSD,not a full Isaac Lab3.0/Sim6.2
installation. No changes to RSI,other environments or the system driver are needed.

```bash
cd /home/ningan/RoboticCuttingSim
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-robot.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
```

On this host `.venv` is already installed. `requirements-robot.lock` includes
the base lock plus exact MuJoCo3.12.0/MuJoCo-Warp3.12.0/schema dependencies.
`requirements.lock` alone suffices for the old blade-only Newton probe. CPU CI
uses only core dependencies,pytest and usd-core; it runs without Newton/Warp/CUDA.

Run the new actual articulated environment and benchmark:

```bash
.venv/bin/python -m robotic_cutting_sim.cli research-demo \
  --device cuda:0 --output-dir artifacts/my_live_cutting
.venv/bin/python -m robotic_cutting_sim.cli benchmark \
  --variants D0 D1 D2 D3 --seeds 42 --device cuda:0 \
  --output-dir artifacts/my_benchmark
```

The first workcell build downloads about21.4 MB of official USD geometry into
ignored `artifacts/workcell_asset_cache`,preserving the official hierarchy and
recording hashes. Native robot/sensor assets are not redistributed. A complete
local hierarchy may be supplied to `scene --preset workcell --asset-root PATH`.
All output paths must be fresh. Live400-step capture takes about165 s on this
host,including full scene ray tracing; it is not realtime. See
[RESEARCH_ENVIRONMENT](RESEARCH_ENVIRONMENT.md) and [BENCHMARK_REPORT](BENCHMARK_REPORT.md).

## Independent NVIDIA RTX capture

OVRTX and OVStage must remain separate from the physics process/runtime:

```bash
python3.12 -m venv .deps/rtx-venv
.deps/rtx-venv/bin/python -m pip install -c requirements.lock \
  -e . Pillow usd-core -r requirements-rtx.lock
```

The binary URLs/hashes are pinned. This downloads roughly1.4 GB and was actually
tested on the current R570/RTX5090 host. Prepare/read USD in a subprocess to
avoid loading external OpenUSD and bundled renderer USD together. Do not remap
CUDA device numbering for the RTX command; use its explicit `--gpu` selection.

For a physics run containing `dual_franka_replay.usda`:

```bash
.venv/bin/python scripts/export_cutting_replay.py \
  artifacts/my_benchmark/D0_seed42/physics \
  artifacts/my_benchmark/workcell/workcell.usda artifacts/my_surface \
  --robot-replay artifacts/my_benchmark/D0_seed42/physics/dual_franka_replay.usda
.deps/rtx-venv/bin/python scripts/capture_ovrtx_rgbd.py \
  artifacts/my_surface/cutting_workcell_replay.usda artifacts/my_rtx \
  --gpu 0 --samples 26
```

This is real RTX capture of measured source states,not live Isaac Kit scheduling.
Native `--camera-mode simulated_d455` captures SingleView postprocessed arrays;
registration/noise limitations remain explicit and it is not a validated live
D455 fallback. Exact canonical commands and videos are in
[WP-03-cutting](reports/WP-03-cutting.md).

## Source and comparison tools

```bash
.venv/bin/python scripts/fetch_sources.py
bash scripts/build_cressim.sh
.venv/bin/python -m robotic_cutting_sim.cli cutting-probe \
  --case cut_recontact --output-dir artifacts/my_recontact
.venv/bin/python -m robotic_cutting_sim.cli cutting-probe \
  --case supported_cross_cuts --duration 4.5 --output-dir artifacts/my_two_cuts
```

Sources are pinned; CRESSim's default build applies both local patch sets and
runs16 actual-header controls before its CUDA12.8/sm120 build. Corrected numerical
probes pass,while persistent cutting remains unsupported. CRESSim runs in its own
process. The native Newton implicit-MPM baseline remains available through
`physics-probe`; it is distinct from the independent persistent-cut TLMPM solver.

Full Lab/Sim GA remains BLOCKED on this installed stack; details and actual
source-only smoke failures are in [COMPATIBILITY](COMPATIBILITY.md). `doctor` and
`official_smokes.py` return nonzero for that blocker,even though standalone
articulation,MPM and RTX now run. No older Isaac version is silently substituted.

# Setup

The current project runs actual **Isaac Sim6.2.0.0** and **Isaac Lab3.0 GA**
in separate Python3.12 environments. The earlier blanket GA-runtime blocker
is historical and has been superseded by full live episodes. The standalone
Newton/Warp/MuJoCo baseline remains independently available.

Use the unified bootstrap from the repository root:

```bash
bash scripts/bootstrap.sh --runtime base --managed-python
bash scripts/bootstrap.sh --runtime isaac_sim --managed-python --accept-eula
bash scripts/bootstrap.sh --runtime isaac_lab --managed-python
```

Alternatively pass `--python /path/to/python3.12` instead of `--managed-python`.
The installer writes only project-local environments, source checkouts and
caches. It does not install a driver, run apt, mutate another project or share
site-packages. Review NVIDIA's license before requesting the Sim installation.
Existing environments are checked against the pins and skipped without package
changes; a mismatched environment is preserved and installation stops.

| Environment | Purpose | USD provider |
| --- | --- | --- |
|`.venv`|CPU tests, workcell generation, standalone baseline|`usd-core26.8`|
|`.deps/isaacsim-venv`|Actual Kit110.4 / PhysX / native RTX|Sim's bundled USD25.11|
|`.deps/isaaclab-venv`|Actual kit-less Lab / NewtonManager|`usd-exchange3.0.1`|

Never install `usd-core` into either Isaac environment or `usd-exchange` into
Sim. Bootstrap installs the frozen complete package lists without resolving
unused converter dependencies that would add a conflicting USD provider.
Torch2.11/cu128 is the measured recipe on RTX5090 / driver570.211.01; the
system CUDA toolkit is not required for these wheels. No driver is upgraded.

Follow [PORTABLE_DEPLOYMENT](PORTABLE_DEPLOYMENT.md) for the doctor, static
scene rebuild, actual-runtime smokes and complete400-step regression commands,
GPU selection, logs and troubleshooting. Its verification report distinguishes
clean-environment reproduction from independent cross-host testing. Runtime
architecture and retained failures are in [ISAAC_SIM_VALIDATION](ISAAC_SIM_VALIDATION.md),
[ISAAC_LAB_INTEGRATION](ISAAC_LAB_INTEGRATION.md) and [COMPATIBILITY](COMPATIBILITY.md).

## Standalone and optional comparison tools

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m robotic_cutting_sim.cli research-demo \
  --device cuda:0 --output-dir artifacts/standalone_NEW
.venv/bin/python -m robotic_cutting_sim.cli benchmark \
  --variants D0 D1 D2 D3 --seeds 42 --device cuda:0 \
  --output-dir artifacts/standalone_matrix_NEW

# Optional source-only comparisons; unnecessary for live CUT/TRIM deployment.
.venv/bin/python scripts/fetch_sources.py
bash scripts/build_cressim.sh
```

Sources are pinned in `configs/source_refs.json`. CRESSim requires a separate
CUDA/CMake build and is not the persistent-cut TLMPM backend. Preserve its
negative results and do not mistake successful compilation for cutting.

The optional OVRTX replay renderer has its own `.deps/rtx-venv` and
`requirements-rtx.lock`; it is not needed for native Sim capture or Lab's
explicit ideal Warp views. Historical replay instructions and capability
boundaries remain in [WP-03-cutting](reports/WP-03-cutting.md). Remote WebRTC
setup is separately optional: [REMOTE_INTERACTIVE_VIEWER](REMOTE_INTERACTIVE_VIEWER.md).

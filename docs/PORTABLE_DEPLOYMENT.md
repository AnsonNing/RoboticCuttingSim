# WP-07: Portable Deployment & Reproducibility

This workflow targets Ubuntu22.04/24.04 x86_64 with an NVIDIA RTX GPU and a
compatible installed driver. It rebuilds the current dual-Franka CUT-01 and
TRIM-01 scene from source and official cached assets. It preserves the existing
physics, headless launchers, D0–D3 evidence and strict TRIM surface failure.
Docker, WebRTC, policy training and driver installation are not dependencies.

## Install

```bash
git clone https://github.com/AnsonNing/RoboticCuttingSim.git
cd RoboticCuttingSim

# Separate steps are useful when only one runtime is needed.
bash scripts/bootstrap.sh --runtime base --managed-python
bash scripts/bootstrap.sh --runtime isaac_sim --managed-python --accept-eula
bash scripts/bootstrap.sh --runtime isaac_lab --managed-python

# Equivalent combined installation.
# bash scripts/bootstrap.sh --runtime all --managed-python --accept-eula
```

Git, Python3.10+ for the bootstrap itself, HTTPS access and an existing compatible
NVIDIA driver are needed. `--managed-python` installs project-local CPython3.12.12
when no suitable3.12 interpreter exists. `--python /path/to/python3.12` uses an
existing interpreter to create independent venvs without inheriting packages.
The bootstrap does not run sudo/apt, change a driver or alter global Python/npm.
`--accept-eula` records acceptance of NVIDIA's Omniverse terms for Sim.

The base, Sim and Lab runtimes use three distinct environments. Paths can be
changed with `--base-env`, `--sim-env`, `--lab-env` and `--lab-source`; defaults
are `.venv`, `.deps/isaacsim-venv`, `.deps/isaaclab-venv`, `.deps/IsaacLab`.
Use the same explicit paths in doctor/validation commands when overriding.
Large-wheel HTTP timeout and download concurrency can be configured with
`UV_HTTP_TIMEOUT` (default900s) and `UV_CONCURRENT_DOWNLOADS` (default4). Failed download/install logs
are retained. Retry a partial installation in new environment paths or a new
clone; a partial existing environment is intentionally preserved and rejected.
Existing environments are inspected before installation and skipped only if
compatible. A mismatch stops without overwriting them. Use a new environment
path for another attempt; preserve failed receipts.

Downloads are cached under `.deps/uv-cache`. `--cache-dir /large/disk/cache`
changes this without changing runtime paths. Put cache and venvs on the same
filesystem for uv hardlinks to deduplicate large Torch/CUDA wheels. Budget at
least30GB free for an initial all-runtime install plus recordings; actual
requirements depend on existing download/Kit caches. A full episode records
material states and three RGB-D streams, so allow additional free disk space.
No original research assets are removed automatically to recover space.

Installation receipts and command logs go into fresh `artifacts/bootstrap/`
subdirectories; `--output-dir` selects an explicit fresh receipt directory.
`--dry-run` prints paths without installation. `--check-only` validates already
installed metadata and never substitutes for a GPU smoke.

## Frozen sources, packages and assets

| Component | Pinned recipe |
| --- | --- |
|Python|3.12; managed default3.12.12|
|Isaac Sim|6.2.0.0, official NVIDIA Python wheels / Kit110.4|
|Isaac Lab|v3.0.0 source `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`|
|Torch|2.11.0+cu128; separate complete CUDA wheel pins|
|Newton / Warp|1.6.1 /1.17.0|
|USD|base `usd-core26.8`; Sim bundled25.11; Lab `usd-exchange3.0.1`|
|Sources/assets/bootstrap downloads|`configs/source_refs.json`, `runtime_assets.lock.json`, `bootstrap_refs.json`|

`requirements-robot.lock`, `requirements-isaac-sim.lock`,
`requirements-isaac-lab.lock` and `requirements-torch-cu128.lock` preserve the
measured environments. Lab's distribution versions differ from its source
release number and are recorded independently. Optional `--fetch-sources`
downloads pinned comparison repositories; it is not required for the runtime.

Bootstrap downloads13 SHA256-checked official Franka/D455 USD composition files
(about21.4MB) into ignored `artifacts/workcell_asset_cache`. It also downloads
and checks the exact official Vulkan shadercache archive needed by Kit; use
`--skip-shadercache` only to let Kit fetch it itself. Source and USD scenes are
regenerated in the new checkout, with new local asset references. Official USD,
venvs, wheels and runtime caches are never committed. Some official MDL material
packages are outside this minimal asset closure; their warnings are retained,
and gray fallback appearance is not claimed as complete material fidelity.

## Doctor and validation

Run the read-only doctor before launching GPU work:

```bash
python3 scripts/deployment_doctor.py --gpu auto \
  --output-dir artifacts/doctor_NEW

# Rebuild static USD and inspect actual local references.
python3 scripts/run_deployment_validation.py --tier static --gpu auto \
  --output-dir artifacts/static_NEW

# Actual two-step episodes in both installed runtimes, with reset/RGB-D checks.
python3 scripts/run_deployment_validation.py --tier smoke --runtime both \
  --gpu auto --output-dir artifacts/smoke_NEW

# Full400-step CUT and2mm TRIM episodes, separately in each runtime.
python3 scripts/run_deployment_validation.py --tier full --runtime both \
  --gpu auto --output-dir artifacts/full_NEW
```

All output directories must be new. Each actual command, return code, duration,
source/configuration hash and run directory is retained. The doctor checks OS,
Python/runtime metadata, package/provider conflicts, GPU/VRAM/driver, CUDA
capability, free disk, extensions and asset caches. A missing `nvcc` is diagnostic;
the pinned CUDA wheels do not require a system development toolkit. Runtime
metadata checks are separate from actually launching Kit or NewtonManager.

`--gpu auto` selects a suitable available physical NVIDIA GPU; `--gpu 0` selects
the only card on a single-GPU host. No command assumes `cuda:1`. The validation
runner passes the selected physical index consistently to Kit, Lab and TLMPM.
Do not remap `CUDA_VISIBLE_DEVICES`: CUDA/renderer device-numbering mismatches
are diagnosed instead of silently rendering on a different card. Available
VRAM is measured at launch; concurrent research jobs can change it.

Smoke PASS requires actual articulation, physical steps, both wrists and Global
RGB-D, synchronized timestamps and reset evidence. It does not imply full
cutting success. Full CUT must retain permanent separation and independent Task
PASS. Full TRIM must retain the separated chip while its strict surface metrics
remain FAIL. The optional `--allow-expected-trim-geometry-fail` accepts only the
known strict surface failure for orchestration; it never relabels that Task PASS
or ignores failed separation, mass conservation, history or camera checks.
Without the flag a strict TRIM failure deliberately produces a nonzero result.

## Encode recorded views without another simulator

The full recorder retains raw RGB-D arrays and commanded/measured traces.
Encode timestamped RGB/depth videos and transform plots after a run:

```bash
.deps/isaaclab-venv/bin/python scripts/render_wrist_camera_evidence.py \
  artifacts/full_NEW/isaac_sim_full_cut artifacts/cut_media_NEW
```

This reads captured arrays; it never launches physics, Kit, a renderer or USD
replay. When system FFmpeg is absent, the encoder uses the Lab environment's
pinned `imageio-ffmpeg` binary. An actual test with `PATH=/nonexistent` encoded
all seven videos successfully. Raw arrays and invalid pixels remain unchanged.

## Troubleshooting and evidence boundaries

- **Wrong Python or existing venv mismatch:** use Python3.12 and a new isolated
  venv path. Do not repair the environment by mixing USD providers.
- **`usd-core` / `usd-exchange` conflict:** use the correct runtime and its lock.
  Sim must use bundled USD, not either external provider. No dependency resolver
  should add unused URDF/MuJoCo converter providers to Sim.
- **`pip check` reports optional dependencies:** the validated Sim recipe retains
  unused converter/OSQP metadata gaps; kit-less Lab retains an unused OVStage
  gap. Exact known lines are WARN with raw nonzero output preserved. Unknown or
  required-package failures remain separate; metadata warnings are not pip-check
  PASS.
- **Driver/CUDA launch failure:** retain the doctor and actual startup log as
  BLOCKED. This installer never upgrades the driver. The original R570/cu128
  execution is a measured host result, not a universal driver certification.
- **Cold stage startup:** the native runner waits for actual stage readiness
  with a300s wall-clock deadline and a short event-loop yield. It writes
  per-reset `stage_load_NN.json`; the old300-update cap was insufficient in
  a fresh clone. A stage-loaded receipt alone is not a physics/camera PASS.
- **Kit extension download stalls:** check HTTPS/CDN access, disk and the pinned
  shadercache receipt. User-level Kit caches may still be used by Kit. Reused
  caches must be disclosed in reproduction reports.
- **Disk exhaustion:** choose an adequately sized filesystem for the checkout,
  environments, same-filesystem cache and recordings. Failed install/run logs
  are preserved; no unrelated data is deleted.
- **TRIM returns FAIL:** inspect separation and surface scoring independently.
  The verified2mm recipe removes98.5294% target material, but strict full-surface
  accuracy remains a known negative result. Never equate exit0 with Task success.

The [deployment validation report](PORTABLE_DEPLOYMENT_VALIDATION.md) records
what was actually installed and executed. A fresh clone with fresh venvs on the
original GPU host is **clean-environment verification**, even when wheel/Kit
caches are reused; it is not independent cross-host reproduction. Another
researcher's Ubuntu/RTX execution remains NOT_RUN until measured.

Official installation references checked for this recipe:
[Sim6.2 Python installation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html),
[Sim6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
and [Lab3.0 source](https://github.com/isaac-sim/IsaacLab/tree/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4).

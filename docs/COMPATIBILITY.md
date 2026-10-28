# Compatibility audit — 2026-10-28

Official references checked before implementation:
[Isaac Lab 3.0 GA](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0),
[Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html),
[Newton 1.6.1 source](https://github.com/newton-physics/newton/tree/v1.6.1),
[CRESSim source](https://github.com/yafei-ou/CRESSim-MPM).

| Item | Actual host / decision |
|---|---|
| OS | Ubuntu 22.04.5, kernel 6.8.0-138 x86_64; supported but deprecated in Sim 6.2 |
| CPU | Threadripper PRO 9975WX, 32 cores / 64 threads |
| RAM | 62 GiB usable, about 53 GiB available at initial audit |
| GPUs | 2 × RTX 5090, 32,607 MiB each; initially no compute jobs |
| Driver | 570.211.01, advertises CUDA 12.8 |
| Local toolkit | CUDA 12.8.93, CMake 3.22.1, GCC 11.4 |
| Disk free | Initially root 69 GiB, /mnt/data 27 GiB; no existing research assets removed |
| Project Python | 3.12.12, own `.venv`; system Python 3.10 is not used |
| Official requested GA | Lab v3.0.0, Sim 6.2.0, Torch 2.11 cu130, Warp 1.17.0, Newton 1.6.1 |
| Installed standalone probe | Newton 1.6.1, Warp 1.17.0, CUDA 12.9 bundled runtime; GPU smoke actually passed on R570 |
| Actual Lab GA runtime | PASS_EXECUTION: pinned v3.0.0 Scene/DirectRLEnv/NewtonManager, Torch2.11+cu128, 400 steps / 8 s live articulated D0 cutting |
| Actual Sim 6.2 / Kit | PASS_EXECUTION: 6.2.0.0 / Kit110.4, native PhysX + native SingleView/RTX, 400 control / 4000 rigid steps / 8 s live D0 |
| Standalone articulation | PASS: Newton + MuJoCo3.12.0 CPU; dual Panda IK/dynamics/collisions executed |
| Standalone RTX | PASS: OVRTX0.6.0/OVStage0.3.0 in `.deps/rtx-venv`; actual RGB-D cutting capture on unchanged R570 |

The previous blanket runtime blocker is superseded by actual execution. The
[official 6.2 Python installation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html)
also supports Torch2.11/cu128. That exact variant allocates and computes CUDA
tensors on both RTX5090s with unchanged driver570.211.01. Actual Lab and full
Kit episodes then executed successfully. The default cu130 stack still has a
different driver requirement; 595.58.03 is the Sim tested recommendation, not
proof of the lowest runnable driver. A DLSS recommendation/preset fallback
warning did not block native Kit capture. No driver verification was disabled.
No system driver, OS, existing venv, RSI directory, or other research project changed.

Source SHAs: [source_refs.json](../configs/source_refs.json). Installed standalone
versions: [requirements.lock](../requirements.lock). Machine-readable raw audit:
`artifacts/environment/doctor/system_report.json` (local, ignored).

Four actual source-only Lab CLI smoke attempts are in `artifacts/official_lab_001`.
They stop at missing GA dependencies (`lazy_loader`) before physics launch; this
was **BLOCKED**, not an official MPM runtime failure or successful test. The
full pinned GA dependencies were subsequently installed in separate
`.deps/isaaclab-venv` and `.deps/isaacsim-venv`; no 2.x substitution was used.

The original standalone research baseline remains available: official
USD geometry,actual articulated control,persistent-cut Warp TLMPM,live full-scene
ideal RGB-D and separately captured native RTX arrays. The
[OVRTX driver table](https://nvidia-omniverse.github.io/ovrtx/driver_requirements.html)
lists a Blackwell R570 baseline; actual capture succeeded here. A recorded R570
denoiser quality/fallback warning does not prevent array capture. This is distinct
from both CUDA13 and the newly executed full Isaac runtime. No driver changed.

OVRTX bundles its own USD/runtime and stays isolated from the physics environment.
Native SingleView source correspondence now passes separately frozen held-out
known-geometry controls in actual OVRTX and actual Kit. Hardware/noise
calibration remains provisional. See [CAMERA_VALIDATION](CAMERA_VALIDATION.md),
[ISAAC_LAB_INTEGRATION](ISAAC_LAB_INTEGRATION.md), [ISAAC_SIM_VALIDATION](ISAAC_SIM_VALIDATION.md),
[WP-03-next](reports/WP-03-next.md) and
[WP-03-cutting](reports/WP-03-cutting.md) for exact receipts and limits.

## Portable installation

The active bootstrap uses the actually measured cu128 recipe with separate
base/Sim/Lab environments and pinned sources. The new read-only
`scripts/deployment_doctor.py` replaces the original WP-00 audit as the current
deployment preflight; the old audit's cu130 gate is historical and is not a
cu128 execution verdict. [PORTABLE_DEPLOYMENT](PORTABLE_DEPLOYMENT.md) explains
installation and actual runtime checks. Clean-clone and independent cross-host
claims are recorded separately in its validation report.

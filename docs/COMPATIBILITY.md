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
| Full GA / Isaac RTX | BLOCKED: CUDA 13 driver gate; Sim official tested Linux driver 595.58.03. Sim/Torch GA stack not installed. |

The official tested driver is not proof that every lower driver fails. R570 does
not satisfy CUDA 13's driver generation, while the separate Warp 1.17 CUDA 12
runtime executed successfully. These are distinct compatibility results.
No system driver, OS, existing venv, RSI directory, or other research project changed.

Source SHAs: [source_refs.json](../configs/source_refs.json). Installed standalone
versions: [requirements.lock](../requirements.lock). Machine-readable raw audit:
`artifacts/environment/doctor/system_report.json` (local, ignored).

Four actual source-only Lab CLI smoke attempts are in `artifacts/official_lab_001`.
They stop at missing GA dependencies (`lazy_loader`) before physics launch; this
is **BLOCKED**, not an official MPM runtime failure or successful test. Install the
full upstream GA environment only on a compatible driver; do not silently use 2.x.

The usable research baseline is independent of that blocked Kit stack: a locally
composed13-file official dual-Panda/dual-D455 USD, a working live Newton+ideal-RGB-D
environment, and combined measured-state replay. Static composition and facade
integration passed; actual RTX sensor/articulation execution remains unvalidated.

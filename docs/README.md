# Documentation

## Start here

| Need | Read |
| --- | --- |
| Install, select a GPU and reproduce | [Portable deployment](PORTABLE_DEPLOYMENT.md) · [Setup](SETUP.md) |
| Run physical-grasp CUT/TRIM | [Physical tool use](PHYSICAL_TOOL.md) |
| Control or connect a research adapter | [Shared environment API](SHARED_TASK_API.md) |
| Use the same-instance remote GUI | [Remote interactive viewer](REMOTE_INTERACTIVE_VIEWER.md) |
| Check current status and unresolved work | [Progress](PROGRESS.md) |

## Current scene and validation

- [Independent physical-tool validation](PHYSICAL_TOOL_INDEPENDENT_VALIDATION.md):
  task, contact, reaction, camera and numerical results, with acceptance limits.
- [Grip perturbation controls](PHYSICAL_TOOL_CONTROLS.md) and
  [native contact sensor](PHYSICAL_CONTACTS.md): actual loads, slip and release.
- [Camera rig](WRIST_CAMERAS.md) and [close Global view](GLOBAL_CLOSE_VIEW.md):
  measured wrist mounts, sensor models and known occlusion limitations.
- [Task definitions](CUTTING_TASKS.md) and [benchmark](CUTTING_BENCHMARK.md):
  workpiece-local goals and independent success criteria.
- [Latest physical-grasp RGB-D recordings](media/wp08_physical_grasp_v2/README.md):
  videos and source hashes for both fixed tasks.

## Runtime and reproduction references

[Compatibility](COMPATIBILITY.md) · [Native Sim integration](ISAAC_SIM_VALIDATION.md) ·
[Lab integration](ISAAC_LAB_INTEGRATION.md) ·
[WP-07 clean-environment results](PORTABLE_DEPLOYMENT_VALIDATION.md)

Current reproducibility checks distinguish new isolated environments on the
original host from independent cross-host tests. A smoke is never full cutting
success; executing an episode is never complete physics acceptance.

## Preserved research history

Earlier reports and media remain in their original paths. They are retained
as evidence and regression baselines, rather than repeated on the front page.

- [Original D0–D3 benchmark](BENCHMARK_REPORT.md).
- [TLMPM validation](reports/TLMPM_VALIDATION.md).
- [Earlier wrist-camera validation](WRIST_CAMERA_VALIDATION.md).
- [Original live-Isaac acceptance](reports/ISAAC_LIVE_ACCEPTANCE.md).
- [WP-08 specification](WP08_SPECIFICATION.md) and
  [frozen baseline](evidence/wp08_baseline.json).

Machine-readable summaries are in [evidence](evidence/); captured media and
their provenance are in [media](media/). Large raw experiments, environments
and official assets remain outside Git in ignored local directories.

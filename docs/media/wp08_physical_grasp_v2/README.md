# Actual physical-tool RGB-D recordings

These are the full 41 captured frames of two eight-second Isaac Sim 6.2
episodes, played at their recorded 5 Hz cadence. The tool is an independent
rigid body supported by finite finger contacts. Encoding invokes neither a
simulation nor a renderer. Original RGB/depth arrays remain unchanged; invalid
depth stays black. The source paths and every input/output SHA256 are in the
adjacent provenance files and [manifest](manifest.json).

| Episode | Combined RGB/depth | Global RGB | Frozen task score |
|---|---|---|---|
| CUT-01 | [Three views](cut01/synchronized_multiview.mp4) | [Overview](cut01/global_d455.mp4) | PASS_TASK_METRICS |
| TRIM-01 | [Three views](trim01/synchronized_multiview.mp4) | [Overview](trim01/global_d455.mp4) | FAIL_TASK_METRICS |

Each directory also contains the individual RGB and optical-Z depth videos for
both wrists and Global. The wrists are `ideal_d405_prototype` geometric cameras;
Global uses the provisional native SingleView D455 pipeline. These videos do
not establish calibrated hardware behavior, complete camera visibility,
coupling-energy convergence or full physical acceptance. TRIM's persistent
separation does not satisfy its original surface-quality criteria. Earlier
failed episodes and contact-timing classifications remain in their original
artifact directories.

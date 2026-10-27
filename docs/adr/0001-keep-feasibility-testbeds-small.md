# ADR 0001 — Blade-only feasibility before robot integration

The restriction on static workcell/API integration is superseded by ADR0002,
following the user's later priority clarification. Scientific evidence boundaries remain.

Accepted2026-10-28. Driver570 supports the tested standalone CUDA12 Warp path,
but the requested GA stack uses CUDA13 and Isaac Sim6.2 is unvalidated. Native
solver persistent separation also remains open.

Use one small Python testbed, a deterministic CSV trajectory, a process-isolated
CRESSim native candidate and an ideal programmable camera. Keep the official
IsaacLab/Newton source pins and record the standalone/runtime boundary explicitly.
Use a static USD for scene inspection and separately labeled measured-state USD
replay. Export raw physics and RGB-D into separate artifact paths.

Do not create a general solver framework, training pipeline, robot controller or
workcell integration before their gates. Do not silently downgrade Isaac or change
system drivers. Do not use geometry removal or visible gaps as fracture proof.
Neither solver qualifies for formal selection; Newton remains the functioning
contact/deformation scaffold while a narrow persistent-separation problem is studied.

Consequences: a useful local material scene and numeric camera PoC can be shared
now. Full dual-Franka/D455 operation, chip dynamics and research-ready benchmark
remain blocked or pending, and reports preserve failed candidates and pre-fix data.

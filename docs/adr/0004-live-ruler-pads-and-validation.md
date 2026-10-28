# ADR 0004 — live ruler integration, pad contact and validation scope

Date: 2026-10-29. Status: implementation accepted; individual live results in
WP-10B/C reports. Historical configurations and measurements remain immutable.

Both live backends accept the three profile tasks through the existing task
factory. Ruler tasks require `physical_grasp`; no fixed ruler is substituted.
The v5 camera rig already has task-independent fixed hand-local mounts and
therefore accepts these task IDs without new aiming or a rig revision.

Finger obstacles come from the two CollisionAPI Cube pads in the composed
official Franka USD, including authored scale and pad-to-body transforms.
Measured current finger bodies determine their edge-frame conservative AABBs
at each 20 ms coupling boundary. The handle is a separate blunt box. Pad
impulses are measured separately, removed from the free-tool reaction and
applied to the corresponding actual finger bodies. Pads cannot mark or sever
material. This is a conservative geometric and delayed coupling approximation;
it does not model skin, exact rotated-box distance, independent within-step
pad velocity or a converged monolithic rigid/material solve.

The nominal per-finger effort remains 20 N and friction 0.6. The sliding
budget is 24 N, compared with the ruler's measured computed weight reported
in WP-10C and the provisional soft-material sub-newton cutting load. The
torsional budget is `2 * 20 N * mu * mean_pad_radius`, from the same actual
pad-overlap geometry as `finite_pad_patch_v3`. Larger food-material loads or
rapid changes can exceed this budget; failure is measured rather than hidden
by an attachment or altered scoring threshold. The rigid controls inherit
every threshold from their frozen parents in new v5 Native / v6 Lab protocols.

Control fuzzing and grasp stress are separate experiments. The declared
4 mm fixed-tool fuzz isolates the bounded-action/IK interface for 1,200
identical requests per runtime. Separate physical-grasp stress pilots expose
Lab dropping under accumulated rotation; those terminations are failures of
holding ability and remain visible. Formal ruler task validation uses actual
physical grasp and the 1 mm task resolution.

State mode no longer performs per-step visual surface reconstruction. Scoring
explicitly refreshes the surface from current physical state. Camera-enabled
runs keep full current RGB-D and measured transforms. Neither option changes
the v1 material solver, forces, task thresholds or existing default cameras.

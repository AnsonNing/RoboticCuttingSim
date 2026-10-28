# Independent physical-tool evidence

The actual Native and Lab CUT-01 recordings pass their unchanged task metrics.
Both TRIM-01 recordings physically separate a chip but fail the original
surface-quality criteria. These are seed 42 engineering results with provisional
material/contact parameters. Complete WP08 physics acceptance remains false.

The [machine-readable receipt](evidence/physical_tool_independent.json) contains
recording paths, protocol/source/input hashes, recomputed metrics, original
failures and separate acceptance scopes. All analysis is CPU-only and reads
immutable recordings. It neither advances a solver nor changes particles,
labels, forces, thresholds or camera arrays.

| Actual full recording | Persistent components / reference masses | Original task score | Camera visibility/optics screen |
|---|---|---|---|
| Native `native_physical_cut01_v2_001` | 1→2; 72g +72g | PASS | PASS |
| Native `native_physical_trim01_v2_001` | 1→2; 1.608g +144.024g | FAIL | FAIL: second wrist loses blade during withdrawal |
| Lab `wp08_lab_physical_cut_v2_001` | 1→2; 72g +72g | PASS | PASS |
| Lab `wp08_lab_physical_trim_v2_001` | 1→2; 1.608g +144.024g | FAIL | FAIL: second wrist loses blade during withdrawal |

Each episode contains 400 control intervals over 8 s and 4,000 formal rigid
substeps. The independent graph includes positive reference-stencil weights,
nonzero stress gradients and code-zero particles. Saved separated topology
persists 2.4 s; immutable reference mass and particle counts agree with the task
sampler. Recomputed task checks and numerical metrics exactly match all four
recorded scores. No future component labels define the rendered partition.

TRIM's detached initial-above-plane fraction is 98.5294%, while Native/Lab
surface p95 errors are 1.6679/1.6642 mm and RMS errors 0.5600/0.6386 mm. Their
strict failures include surface error, gouging and overcut volume. Separation
does not establish successful flattening. The earlier Native CUT with a 4 mm
reference-interface error also remains a failed task.

The tool is a free dynamic USD body outside the hand hierarchy, with no
attachment joint, positive actual mass/inertia and gravity enabled. Its
handle, connected shaft and blade own their collision geometry. Actual body
state composes the measured edge and matches the material adapter input.
Applied edge forces are shifted to the actual pre-step COM and sent to the
independent body. Structural, archived-source, state and wrench-routing
checks pass for all four full recordings. These checks alone do not establish
a robust grasp or calibrated mechanics.

The candidate uses finite 20 N per finger effort limits, friction 0.6, position
stiffness 2,000 N/m and damping 20 Ns/m. The explicit candidate initial gap is 11.2 mm;
the earlier 14 mm initialization and its failures remain preserved. Native's
explicit contact geometry equals the actual pre-step pose, its timestamp
equals current time minus 2 ms, and the unchanged 1 µm handle-region check passes
for all 4,000 formal steps of each candidate episode, with 100% loaded bilateral
handle contact.

Earlier Native full recordings classified contact points using the integrated
post-step pose, producing 91.35%/92.85% fractions. The read-only timing
diagnostic retains those failures. Applying the identical region and load
limits to their recorded pre-step pose yields 100%/100%, with maximum handle
boundary excess below 9 nm. This is a measured discrete-backend timing result,
consistent with PhysX's distinction between event geometry and subsequent
[actor state readback](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Simulation.html#callback-sequence).
It is not a universal CCD or extrapolated-pose rule.

Native smoke002 is explicitly rejected as a previous false positive. Its
Python flags stayed fresh while C++ reported 242 invalidated-view lines in
stdout, repeated in the Kit log. Removing old rigid camera assets after
physics initialization invalidated cached tensor views. Only its preceding
pregrasp evidence remains usable. New candidate episodes author the rig before
physics views and contain zero known invalidation lines; all formal contact
steps are fresh. The interrupted first full Native episode is also preserved.

The independent seven-case controls reader reconstructs normal load from
actual matched contact pairs and impulses, tool/hand relative motion, raw
penetration and unsupported COM motion. It reproduces the original Native v1
strict failure and both Lab v1/v2 incomplete matrices. Native v2 passes the
frozen engineering screens: the strong 1 N case drifts 0.689 µm with 100% bilateral
handle contact, while matched weak/low-friction cases move approximately
265.77/265.79 mm. Release contains 192 ms of unsupported motion with measured
vertical acceleration −9.80998967 m/s². Known load/torque produces actual arm
response. Stress/drop cases retain their complete data; a screen requiring
finite response does not claim that the tool stayed held.

A separate actual Native feedback ON/OFF pair has identical sources, initial
material, initial measured state and desired actions; only feedback changes.
OFF applies zero COM wrench. ON's material force peaks 0.272892 N. Maximum
measured differences are 4.76837e−5 rad in the arm,45.884 µm in the free tool and
0.287111 N in finger contact force. This passes the prefrozen response screen,
with the 20 ms feedback delay explicit. Native has recipient/source evidence,
but no independently recorded numeric pre-step hand external-force buffer
for these episodes. Full two-way physical acceptance is not inferred.

The rigid-tool impulse audit was frozen before force/velocity residuals were
read: commit`b2fa1e8`, [protocol](../configs/tool_momentum_audit_v1.json). It
integrates normal and friction forces, applied COM wrench and gravity, and
compares actual COM momentum changes. Angular momentum uses
`R I_body Rᵀ ω`, consistent with the documented
[body-frame COM inertia convention](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html).
Contact torques use each recorded world point and pre-step COM. No sign,
lever-arm, frame or scale is fitted to reduce residuals.

The frozen per-step engineering budget is an absolute term plus 5% of the
sum of constituent impulse norms, before cancellation. Absolute terms are
2e−5 N·s and 1e−6 N·m·s; every row must pass. All residual tails and accumulated
vector errors remain in the output.

| Recording | Maximum linear residual (N·s) | Maximum angular residual (N·m·s) | Important limit |
|---|---:|---:|---|
| Native CUT v2 | 7.51e−9 | 3.84e−6 | Angular absolute term exceeded in all 4,000 steps |
| Native TRIM v2 | 5.46e−9 | 2.27e−6 | Angular absolute term exceeded in all 4,000 steps |
| Lab CUT v2 | 1.65e−7, assumed-gravity diagnostic | 5.66e−9 | Linear NOT_SCORED: rigid gravity readback absent |
| Lab TRIM v2 | 9.05e−8, assumed-gravity diagnostic | 3.09e−9 | Linear NOT_SCORED: rigid gravity readback absent |

Native passes the declared relative engineering budget, but does **not**
establish complete angular closure: accumulated Z residuals are 0.0117622 and
0.00747527 N·m·s. Full inertia readback agrees with the tool recipe. The exact
constraint/friction-anchor moment bookkeeping remains unresolved. The
absolute-only diagnostic fails and is reported beside the original formula,
without replacing it or changing the physics. Lab's later v3 two-step smoke
records actual Newton/native gravity and pre-step hand-force buffers; its 20
formal rows pass the same linear/angular budget. That short zero-material-load
check does not validate a full v3 cutting episode.

Rigid impulse bookkeeping also does not establish material-interface energy
convergence. Measured interface-work residuals are 16.80%/15.92% for Native
CUT/TRIM and 18.83%/11.49% for Lab. All fail the unchanged 5% `PHY-04` screen in
[isaac_acceptance.json](../configs/isaac_acceptance.json).

Camera analysis uses the complete current scene, keeps handle/shaft occlusion
separate from blade and material, and preserves native depth holes/outliers.
All four episodes have 41 recorded sample times and three RGB-D streams;
maximum recorded robot/material/surface/calibration timestamp disagreement is
0 s. This is recorded-clock alignment, not an independently measured hardware
exposure clock. Native active-cut blade pixel minima are 2621/696/94 for CUT
and 1582/663/61 for TRIM (wrist0/wrist1/Global). Both mechanical components have
minimal actual depth-consistent pixel evidence after separation. TRIM's
second-wrist withdrawal occlusion remains a strict full-episode failure.
The D405s remain ideal geometric prototypes; Global D455 registration is
version-gated and provisional. Hardware calibration and complete sensor
acceptance are not claimed.

Actual recordings are available as [CUT RGB-D](media/wp08_physical_grasp_v2/cut01/synchronized_multiview.mp4)
and [TRIM RGB-D](media/wp08_physical_grasp_v2/trim01/synchronized_multiview.mp4).
Each task also has all three individual RGB/depth videos. All 14 videos fully
decode, retain their 41 input samples and copy byte-for-byte to the public
media directory; the [manifest](media/wp08_physical_grasp_v2/manifest.json)
records exact output hashes and source provenance. Encoding invokes no
simulation or renderer.

Representative reproduction commands, using fresh output directories:

```bash
.venv/bin/python scripts/audit_physical_tool.py EPISODE artifacts/new_tool_audit
.venv/bin/python scripts/audit_tool_momentum.py EPISODE artifacts/new_momentum_audit
.venv/bin/python scripts/audit_physical_controls.py MATRIX/index.json artifacts/new_control_audit
.venv/bin/python scripts/audit_wrist_camera_rig.py --episode EPISODE --output-dir artifacts/new_camera_audit
.venv/bin/python scripts/audit_cutting_task_episodes.py --recording-dir EPISODE --output-dir artifacts/new_task_audit
.venv/bin/python scripts/compare_material_feedback.py --on ON_EPISODE --off OFF_EPISODE --output-dir artifacts/new_feedback_audit
.venv/bin/python scripts/render_wrist_camera_evidence.py EPISODE artifacts/new_actual_media
```

The focused CPU suite passes 68 tests covering tool geometry/mass/inertia,
capsule initialization, adversarial source/contact auditing, timing, physical
part attribution and analytical impulse/control checks. These generated CPU
fixtures are tests of analysis code, separate from the actual runtime evidence.

The subsequent actual Lab full CUT v3 recording adds independent rigid/native
gravity readback and numeric pre-step hand-force telemetry. All 4,000 formal
rows pass the original impulse budget and the standalone absolute terms:
maximum linear residual 1.65522e−7 N·s and angular residual 4.95679e−9 N·m·s.
The actual free-body/source/edge audit also passes. These additional results
do not retrofit missing gravity into v2. The independent v3 rigid-control
matrix still fails four transfers (baseline, nominal load, low friction and
release); general grasp stability across the transfer protocol remains failed.

The optional Native rack007 skill was also independently inspected, including
its composed final stage, archived sources and 20,090 actual rigid steps plus
1,000 excluded settling steps. The end-release finger positions are
39.999984 mm each. Actual positive normal impulses identify the two rack rails
as support during all 1,880 withdrawal steps, while the hand moves 79.944 mm
and the tool moves only 49.4 nm. Reacquisition lifts the actual free body 79.627 mm;
the following 1,000 steps have 100% loaded bilateral handle contact, zero loaded
rack/table/palm support, relative drift 0.531 µm and rotation 11.54 µrad.
The [actual captured keyframes](media/wp08_tool_rack_007/actual_keyframes.png)
show release, withdrawal, closing and holding, with original timestamps.

This is an optional rigid-only Native skill, not a cutting score or Lab pickup
proof. Raw normal reports establish rack contact identity; rack bodies are
absent from the tensor-force filters, so a complete signed rack frictional
wrench remains NOT_SCORED. The driver records a material-unchanged Boolean
and uses the source-audited rigid-only path, but does not serialize its original
before/after material hashes for independent recomputation. Earlier rack001–006
failures remain preserved. Reproduce this analysis with:

```bash
.venv/bin/python scripts/audit_tool_rack.py RACK_EPISODE artifacts/new_rack_audit
```

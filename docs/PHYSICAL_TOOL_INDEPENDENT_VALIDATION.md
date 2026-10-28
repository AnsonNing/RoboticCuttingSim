# Independent physical-tool evidence

The actual Native and Lab CUT-01 recordings pass their unchanged task metrics.
Both TRIM-01 recordings physically separate a chip but fail the original
surface-quality criteria. These are seed42 engineering results with provisional
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

Each episode contains 400 control intervals over8s and4000 formal rigid
substeps. The independent graph includes positive reference-stencil weights,
nonzero stress gradients and code-zero particles. Saved separated topology
persists2.4s; immutable reference mass and particle counts agree with the task
sampler. Recomputed task checks and numerical metrics exactly match all four
recorded scores. No future component labels define the rendered partition.

TRIM's detached initial-above-plane fraction is98.5294%, while Native/Lab
surface p95 errors are1.6679/1.6642mm and RMS errors0.5600/0.6386mm. Their
strict failures include surface error, gouging and overcut volume. Separation
does not establish successful flattening. The earlier Native CUT with a4mm
reference-interface error also remains a failed task.

The tool is a free dynamic USD body outside the hand hierarchy, with no
attachment joint, positive actual mass/inertia and gravity enabled. Its
handle, connected shaft and blade own their collision geometry. Actual body
state composes the measured edge and matches the material adapter input.
Applied edge forces are shifted to the actual pre-step COM and sent to the
independent body. Structural, archived-source, state and wrench-routing
checks pass for all four full recordings. These checks alone do not establish
a robust grasp or calibrated mechanics.

The candidate uses finite20N-per-finger effort limits, friction0.6, position
stiffness2000N/m and damping20Ns/m. The explicit candidate initial gap is11.2mm;
the earlier14mm initialization and its failures remain preserved. Native's
explicit contact geometry equals the actual pre-step pose, its timestamp
equals current time minus2ms, and the unchanged1µm handle-region check passes
for all4000 formal steps of each candidate episode, with100% loaded bilateral
handle contact.

Earlier Native full recordings classified contact points using the integrated
post-step pose, producing91.35%/92.85% fractions. The read-only timing
diagnostic retains those failures. Applying the identical region and load
limits to their recorded pre-step pose yields100%/100%, with maximum handle
boundary excess below9nm. This is a measured discrete-backend timing result,
consistent with PhysX's distinction between event geometry and subsequent
[actor state readback](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Simulation.html#callback-sequence).
It is not a universal CCD or extrapolated-pose rule.

Native smoke002 is explicitly rejected as a previous false positive. Its
Python flags stayed fresh while C++ reported242 invalidated-view lines in
stdout, repeated in the Kit log. Removing old rigid camera assets after
physics initialization invalidated cached tensor views. Only its preceding
pregrasp evidence remains usable. New candidate episodes author the rig before
physics views and contain zero known invalidation lines; all formal contact
steps are fresh. The interrupted first full Native episode is also preserved.

The independent seven-case controls reader reconstructs normal load from
actual matched contact pairs and impulses, tool/hand relative motion, raw
penetration and unsupported COM motion. It reproduces the original Native v1
strict failure and both Lab v1/v2 incomplete matrices. Native v2 passes the
frozen engineering screens: the strong1N case drifts0.689µm with100% bilateral
handle contact, while matched weak/low-friction cases move approximately
265.77/265.79mm. Release contains192ms of unsupported motion with measured
vertical acceleration−9.80998967m/s². Known load/torque produces actual arm
response. Stress/drop cases retain their complete data; a screen requiring
finite response does not claim that the tool stayed held.

A separate actual Native feedback ON/OFF pair has identical sources, initial
material, initial measured state and desired actions; only feedback changes.
OFF applies zero COM wrench. ON's material force peaks0.272892N. Maximum
measured differences are4.76837e−5rad in the arm,45.884µm in the free tool and
0.287111N in finger contact force. This passes the prefrozen response screen,
with the20ms feedback delay explicit. Native has recipient/source evidence,
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

The frozen per-step engineering budget is an absolute term plus5% of the
sum of constituent impulse norms, before cancellation. Absolute terms are
2e−5N·s and1e−6N·m·s; every row must pass. All residual tails and accumulated
vector errors remain in the output.

| Recording | Maximum linear residual (N·s) | Maximum angular residual (N·m·s) | Important limit |
|---|---:|---:|---|
| Native CUT v2 | 7.51e−9 | 3.84e−6 | Angular absolute term exceeded in all4000 steps |
| Native TRIM v2 | 5.46e−9 | 2.27e−6 | Angular absolute term exceeded in all4000 steps |
| Lab CUT v2 | 1.65e−7, assumed-gravity diagnostic | 5.66e−9 | Linear NOT_SCORED: rigid gravity readback absent |
| Lab TRIM v2 | 9.05e−8, assumed-gravity diagnostic | 3.09e−9 | Linear NOT_SCORED: rigid gravity readback absent |

Native passes the declared relative engineering budget, but does **not**
establish complete angular closure: accumulated Z residuals are0.0117622 and
0.00747527N·m·s. Full inertia readback agrees with the tool recipe. The exact
constraint/friction-anchor moment bookkeeping remains unresolved. The
absolute-only diagnostic fails and is reported beside the original formula,
without replacing it or changing the physics. Lab's later v3 two-step smoke
records actual Newton/native gravity and pre-step hand-force buffers; its20
formal rows pass the same linear/angular budget. That short zero-material-load
check does not validate a full v3 cutting episode.

Rigid impulse bookkeeping also does not establish material-interface energy
convergence. Measured interface-work residuals are16.80%/15.92% for Native
CUT/TRIM and18.83%/11.49% for Lab. All fail the unchanged5% `PHY-04` screen in
[isaac_acceptance.json](../configs/isaac_acceptance.json).

Camera analysis uses the complete current scene, keeps handle/shaft occlusion
separate from blade and material, and preserves native depth holes/outliers.
All four episodes have41 recorded sample times and three RGB-D streams;
maximum recorded robot/material/surface/calibration timestamp disagreement is
0s. This is recorded-clock alignment, not an independently measured hardware
exposure clock. Native active-cut blade pixel minima are2621/696/94 for CUT
and1582/663/61 for TRIM (wrist0/wrist1/Global). Both mechanical components have
minimal actual depth-consistent pixel evidence after separation. TRIM's
second-wrist withdrawal occlusion remains a strict full-episode failure.
The D405s remain ideal geometric prototypes; Global D455 registration is
version-gated and provisional. Hardware calibration and complete sensor
acceptance are not claimed.

Actual recordings are available as [CUT RGB-D](media/wp08_physical_grasp_v2/cut01/synchronized_multiview.mp4)
and [TRIM RGB-D](media/wp08_physical_grasp_v2/trim01/synchronized_multiview.mp4).
Each task also has all three individual RGB/depth videos. All14 videos fully
decode, retain their41 input samples and copy byte-for-byte to the public
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

The focused CPU suite passes68 tests covering tool geometry/mass/inertia,
capsule initialization, adversarial source/contact auditing, timing, physical
part attribution and analytical impulse/control checks. These generated CPU
fixtures are tests of analysis code, separate from the actual runtime evidence.

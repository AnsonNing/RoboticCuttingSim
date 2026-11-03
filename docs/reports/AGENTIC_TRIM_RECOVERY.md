# Agentic control robot: Native recovery and continued cutting research

Status: **IN_PROGRESS**. The previous bounded study remains a separate,
unchanged record. Runtime readiness does not complete this continuation;
autonomous cutting, independent physical evaluation across different scenes,
and a true autonomous task success with complete verified media are required.

## Runtime changes and actual validation

The opt-in streamed recorder writes the original reference positions, masses,
stencil nodes, weights and gradients once, then appends sampled time, positions,
velocities and cut codes. Each committed sample is flushed before an atomic
checkpoint update. Read-only memory maps recover the committed prefix after an
interruption; incomplete recordings cannot pass integrity. A subprocess
interruption check and exact legacy-versus-streamed evaluator comparison pass.
The default legacy recorder remains available.

This mode also retains only the latest live material sample, avoiding duplicate
unbounded histories and final stacking. Recorded precision, TLMPM v7_2 physics,
goal thresholds, support-interface connectivity and code-zero bridges are
unchanged. Whole-process Native memory measurements include rendering and
other runtime allocations; they do not isolate the recorder's causal effect.
Host headroom also improved relative to the previous study. Historical cgroup
OOM counters did not identify every victim and remain qualified accordingly.

The versioned `isaac62_physx_offline_v1` profile presents byte-verified official
registry metadata through episode-owned local registries. It enables the
installed registry extension, configures the actual process registry provider,
and activates the installed contact extension. Vendor payloads, manifests,
shared installations and required sensors are preserved. Runtime receipts pin
Isaac Sim 6.2.0.0, Kit 110.4.0 build 187cf21b, physics sensor 4.0.0 and the
original PhysX 110.4.4 dependencies. Sensor activation and actual fresh
post-physics updates are checked separately.

Three preserved startup failures preceded the first complete episode:

| Attempt | Actual evidence | Result |
| --- | --- | --- |
| 001 | Startup completed; primary reset exception was obscured during Kit teardown. Premature registry-module import is a probable cause, not a confirmed original exception. | FAIL initialization; no scored transition |
| 002 | The real offline provider read its official metadata, then a literal URL comparison rejected equivalent local file URLs. Error receipts persisted before teardown. | FAIL initialization; no scored transition |
| 003 | Actual contact extension and native plugin started offline. The fresh frozen checkout lacked the ignored official Franka USD cache. | FAIL asset provisioning; no scored transition |

Semantic local-URL comparison fixed attempt002. The official asset cache was
copied from the preserved previous frozen environment and every file verified
for attempt004. No seed, controller, physics or scoring change accompanied
that retry. Provenance now records the actual frozen solver commit in Native,
and inherited controls retain their original controller commit and parent run.

## Complete scripted feasibility reference

Seed0, with two irregular protrusions, completed **1200 actions / 24 seconds**
under the real offline sandbox and GPU1 device isolation. Frozen runtime source
is `fdaffffb99ffc3433f921fac2741e234cd38ae11`. The original read-only physical
evaluator actually executed both before Native teardown and independently in
the external file-evaluation process; both results were PASS.

| Measurement | Result |
| --- | --- |
| Complete physical stream | 121 samples, 0–24 seconds, 103581 particles; finalized and complete |
| Integrity / mechanical outcome | PASS / PASS |
| Detached target mass | 100% for each protrusion; three mechanical components |
| Detached mass below the plane | 0%; original substrate thresholds pass |
| Surface quality / HOME | Measured separately / NOT_RUN |
| Native contact updates | All 13000 settling/coupled readings valid and fresh, with advancing steps and time |
| Live cameras | 121 synchronized Global RGB-only and wrist_0 RGB/depth/mask pairs; recorded calibration agrees |
| Parked arm_1 | Command unchanged; maximum measured joint error below 1e-10 rad |
| Captured process RSS | 5581384–6403528 KiB; no new observed cgroup OOM counter |
| RSS before/after recording finalization | 5654000 / 5654040 KiB; no final-stack spike at these measured boundaries |
| Full MP4 | Entire decode PASS; frame, physical-recording and source identities PASS |
| Episode wall time | 939.54 seconds, including the independent external evaluation |

The full MP4 SHA256 is
`30c4fca85323b01d70c9ca2290a28770c4493f46d307b05f19b2ca5924ff0601`.
Complete raw frames, physical arrays and failed attempts are retained locally.
This controller is the existing scripted diagnostic and **is excluded from
every autonomous success rate**. Mechanical disconnection does not require a
detached chip to disappear from the camera; images are never the task scorer.
Descriptive surface measurements remain separate: p95 absolute plane error
0.480 mm, reconstructed overcut-volume fraction 1.38%, and maximum gouge-depth
proxy 20 mm. These values do not establish complete surface-quality acceptance.
Material parameters and simulated camera calibration remain provisional and
uncalibrated against physical hardware.

## Autonomous continuation

The exact inherited C2 completed the one-protrusion seed1 through the same
frozen adapter and offline sensor namespace: 1200 explicit actions, including
137 cut-enabled actions, and 24 seconds. Recording integrity and runtime
readiness PASS; mechanical outcome FAIL. Its true detached target fraction is
64.8%, below the unchanged 90% threshold. Substrate retention passes, with no
detached mass below the plane. There were no rejected actions. Complete raw
failure data and the controller's own observations and events are preserved.

The existing researcher continuation has actually started with exact inherited
C1/C2 sources, prior reasoning and the completed C2's own sensor/proprioception
diagnostics plus sanitized independent outcome feedback. It is analyzing its
observations and running paired inherited comparisons on seeds15 and29.
Supervision supplies environment repairs and evaluation, without providing a
successful trajectory or privileged material state.

The first comparison submission failed before Native startup because the
frozen checkout's Git alternates referred outside the research sandbox. Its
objects were made self-contained, the retired alternates file retained, and
the exact source commit/tree verified with the original Git check inside the
actual research namespace. The agent independently prepared a temporary
development-only index check while that repair was in progress; this wrapper
and its limitation are retained. It changes no controller, physics, GPU guard
or external evaluator. The independent canonical source check also passes.

Autonomous improvement and new held-out results are **NOT_RUN** at this
snapshot. No autonomous success is claimed. Continue the agent's own revisions,
freeze a selected controller before fresh held-out seeds, and independently
score every declared scene. Every true success, including development, must
retain its complete verified MP4; all failures remain evidence.

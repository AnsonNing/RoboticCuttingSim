# Native Isaac Sim remote viewer runtime

The remote viewer starts the existing Isaac Sim6.2 Python runtime once, creates
`LiveIsaacSimCutting` in that same process, and streams its Kit application
framebuffer. It does not start an additional simulator or render recorded USD
motion. The sensor render products and observations remain those of the live
cutting environment.

## Runtime and transport

The bootstrap is `robotic_cutting_sim.remote_runtime.launch_streaming_app`;
its experience is `configs/isaac_sim_cutting_streaming.kit`. It retains all13
dependencies of the original controlled cutting experience and adds the Kit
main window, viewport, stage/property panels, toolbar, console and official
`omni.kit.livestream.app` extension. NVIDIA documents this extension as streaming
the entire application framebuffer, including the application UI. The browser
therefore receives the same running scene and GUI.
[NVIDIA application streaming extension](https://docs.omniverse.nvidia.com/kit/docs/omni.kit.livestream.app/latest/Overview.html)

`SimulationApp` uses `headless=True`, `hide_ui=False` and
`disable_viewport_updates=False`. Headless disables a local OS window; explicitly
showing the UI keeps the offscreen GUI available for streaming. This combination
is documented in the installed6.2 `simulation_app.py` configuration API.

The existing installation is Isaac Sim6.2.0.0 / Kit110.4.0. The host driver and
environment are unchanged. Kit may resolve missing streaming/UI extensions from
the NVIDIA110.4 shared registry and the SDK registry for its own build hash.
The bootstrap also adds this project's previously resolved extension-cache folder
to avoid fetching already installed dependencies. It does not install another
Isaac Sim distribution or change the existing headless launch function.

| Endpoint | Default | Owner |
|---|---|---|
| Advertised server IP | `100.111.162.17` | Native streaming bootstrap |
| WebRTC signaling | TCP49100 | Kit WebRTC extension |
| WebRTC media | UDP47998 | Kit WebRTC extension |
| Browser web viewer | TCP8210 | Separate official web-viewer host |

Set `publicIp`, `signalPort`, and `streamPort` under
`/exts/omni.kit.livestream.app/primaryStream`. The client must use the same host
address. The HTTP port is separate from the signaling/media ports; changing one
does not change the others. Both TCP signaling and UDP media must be reachable.
[Isaac Sim6.2 livestream clients](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/manual_livestream_clients.html)

Avoid old `/app/livestream/publicEndpointAddress` and related port settings.
Kit migrates these legacy settings and gives them precedence over the current
settings. A conflicting old value can silently advertise an incorrect endpoint.
[Kit streaming settings migration](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/109.0.6/guide/migration_108.html)

Read-only Tailscale inspection confirmed host`100.111.162.17` and Mac
`100.121.246.92` online with an active connection. No firewall, OS, driver or
Tailscale configuration was changed. An existing Tailscale DNS health warning
does not prevent using the numeric overlay IPs. Browser connectivity and video
decode still require a separate actual client check.

## Authoritative physics stepping

The installed6.2 `SimulationManager.step()` calls the physics simulation interface
directly with the configured physics timestep. It does not require timeline Play.
The existing cutting backend calls this explicit API for its articulation
substeps, reads measured hand/tool poses, then advances the retained TLMPM
material and its contact reaction. A GUI-only PhysX timeline step would bypass
that coupling.

The streaming experience sets `/app/player/playSimulations=false`.
`configure_manual_physics()` additionally disables automatic timeline updates
and the SimulationManager warm-start/on-stop callbacks. Call this helper after
every reset, retaining the stage-open/close callbacks. The helper never calls
timeline Stop or invalidates an initialized articulation. GUI Play/Pause/Reset
events must queue the project's corresponding controlled operations; they must
not enable automatic PhysX playback.

An idle GUI pumps `app.update()` so input, panels and video continue updating.
Only the controlled `LiveIsaacSimCutting.step()` advances simulation. The paused
runtime check must compare physics-step count, physics time, robot joint/hand
state and material state before and after GUI updates/captures. Take the first
comparison after reset initialization: the native SimulationManager initializer
explicitly performs warmup physics steps before the controlled episode starts.

## Safe camera previews

`ViewportCameraPreview` exposes Free, global D455 and both wrist camera choices.
It reads the current optical USD world transform and camera lens into separate
viewer-only cameras under `/World/RemoteViewer`; it never assigns a physical
sensor camera as the editable viewport camera. This prevents streamed orbit or
pan gestures from changing a sensor mount or invalidating its extrinsic checks.

Call `update()` after reset/control steps and before GUI updates. The sensor
preview follows the current measured robot scene; Free retains the user's
navigation. These previews are normal scene viewports, not replacements for
RGB-D observations or native depth validity masks. The actual viewport aspect
ratio can differ from the sensor render product.
[NVIDIA viewport utility API](https://docs.omniverse.nvidia.com/kit/docs/omni.kit.viewport.utility/latest/index.html)

## Verification boundary

Eleven CPU checks pass for endpoint validation, preservation of the original
controlled-physics dependencies and copying a moving hand-child camera without
modifying its physical source. The command is:

```bash
.venv/bin/python -m pytest tests/test_remote_runtime.py -q
```

`streaming_runtime.json` records requested settings, source hashes, actual enabled
extension IDs and actual stream endpoints after startup. An initialized app or a
listening port alone does not establish successful browser interaction. Preserve
the separate startup, paused-state, coupled-step, camera-switch and actual
Mac-client connection receipts.

Actual GPU1 smoke002 initialized the official streaming extension and Kit GUI,
loaded the original TRIM-01 native environment at2mm particle spacing, switched
all four viewport choices, performed two coupled20ms steps and captured both
848×480 wrist RGB-D outputs plus the640×480 global native D455 output. Forty
paused GUI updates and another five updates after capture preserved physics time,
step count, measured robot state and all captured material-state fields exactly.
The committed compact receipt is
[remote_runtime_smoke.json](evidence/remote_runtime_smoke.json); the complete local
logs and arrays remain in `artifacts/remote_runtime_smoke_002`.

The actual enabled extensions were livestream.app10.4.1, livestream.webrtc10.7.4,
viewport.window110.0.5, window.stage2.6.4 and window.property1.15.2. Kit reported
the primary server on49100/47998; TCP49100 was observed listening. The encode
library is installed, but an encoded NVENC frame and a browser connection were
not tested by this smoke. UDP media was not observed bound before any client
connected. These are explicit remaining client-validation boundaries, not an
end-to-end streaming claim.

Smoke001 is retained. Its immediate auto-update readback was stale because the
timeline change was queued; smoke002 explicitly commits that change on the Kit
thread before recording it. Both runs preserved the paused physical state. The
original asset-cache MDL shading warnings remain in the logs; no existing
robot/material physics or asset files were changed for streaming.

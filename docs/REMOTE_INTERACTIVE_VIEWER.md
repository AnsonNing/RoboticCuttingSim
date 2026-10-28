# Remote Interactive Viewer

This optional viewer adds the full Kit GUI to the existing live Isaac Sim6.2
runner. The original isolated Sim environment, PhysX articulations, TLMPM,
workpiece generator, task evaluator and camera recorder are reused. It is not
required by portable deployment or headless benchmarks. Development stays in
VS Code Remote-SSH.

## WP-08 physical-tool viewer

The current live session uses the independent contact-grasped knife, with
Hold/Close/Open/Release controls and slip measured since actual pregrasp
settling. Opening fingers permits a real drop; detected grasp loss pauses
the shared episode until Reset. Tool truth in the GUI/status file is privileged
debug information and does not enter deployable observations.

```bash
# Reuse the installed native runtime and existing HTTP viewer.
OMNI_KIT_ACCEPT_EULA=YES .deps/isaacsim-venv/bin/python scripts/run_remote_viewer.py \
  --task CUT-01 --gpu 0 --tool-attachment-mode physical_grasp \
  --grip-initial-position 0.0112 --output-dir artifacts/physical_viewer_NEW

# Graceful stop of this example session only.
touch artifacts/physical_viewer_NEW/STOP
```

The running original-host instance is currently
`artifacts/remote_interactive_physical_001`, tmux `rcs-wp08-remote`, at
[http://100.111.162.17:8210](http://100.111.162.17:8210).
The previous `remote_interactive_003` legacy instance was gracefully closed,
and its evidence remains unchanged. The HTTP viewer still serves the same
official client and ports; no second simulator or container was introduced.

Actual local Chrome connected through this Tailscale URL and passed15 streamed
transactions: wrist/global selection, step, release, play-to-physical-drop and
automatic termination, pregrasped reset, CUT/TRIM switching, and typed joint
and Cartesian actions. Video RTP decode and advancing1280×800 frames were
measured independently of GUI acknowledgement. See the
[compact receipt](evidence/physical_tool_viewer.json) and
[actual screenshot](media/physical_tool_viewer.png). An external Mac client
remains **NOT_RUN**; this local browser test does not certify it.

## Start

Connect both hosts to the same Tailscale network. On the simulation host, from
this checkout, use two terminals or persistent tmux sessions:

```bash
# One-time client build; project-local Node and pinned official NVIDIA SDK.
bash scripts/setup_remote_web_viewer.sh

# Terminal1: static official-SDK web client; no simulator is launched here.
.venv/bin/python scripts/serve_remote_web_viewer.py

# Terminal2: the existing Sim environment and one live cutting application.
OMNI_KIT_ACCEPT_EULA=YES .deps/isaacsim-venv/bin/python \
  scripts/run_remote_viewer.py --task TRIM-01 --gpu 0 \
  --output-dir artifacts/remote_session_NEW
```

`--gpu 0` works on a single-GPU host. Specify another physical index on a
multi-GPU host without remapping `CUDA_VISIBLE_DEVICES`. Output directories
must be new. Startup writes `status.json`; wait for `phase: READY` before
connecting. No desktop X server or separate simulator/container is needed;
Kit still creates its complete streamed GUI and viewport.

The current host's URL is **http://100.111.162.17:8210**. Open it in Mac Chrome
while connected to Tailscale. Host IP detection uses `tailscale ip -4`; use
`--ip` for the Sim runner and `--host` / `--signal-host` for the web server to
override it together. See [the web-client documentation](REMOTE_WEB_VIEWER.md)
for exact dependencies and client/server options.

| Port | Transport | Purpose |
| --- | --- | --- |
|8210|TCP|Viewer page, bound only to Tailscale IP|
|49100|TCP|Official WebRTC signaling|
|47998|UDP|Official encoded video/media|

Allow these endpoints within the authorized tailnet. No firewall or Tailscale
ACL is changed by the scripts. The SDK's native signaling listener may bind
all interfaces; network exposure remains the administrator's network policy.
Only one browser streaming client should connect at a time.

## Operate the same simulation

Click the streamed **RoboticCuttingSim** panel. CUT-01 / TRIM-01 creates a fresh
material and robot reset in the same application. Play advances the scripted
reference; Pause holds state; Step advances one20ms control interval; Reset
clears cut history and restores the selected task. Task evaluation remains
independent of episode completion. TRIM uses its verified2mm sampling and keeps
its strict surface-accuracy FAIL.

Free, Global D455 and both Wrist D405 controls select the main viewport.
Sensor previews copy the physical camera pose/lens into a separate viewport
camera. Mouse orbiting therefore cannot move the actual wrist mount. Formal
RGB-D always comes from the original named sensor streams.

For an explicit action, choose Cartesian or Joint, populate a current pose
or joints, edit the seven-value JSON array, then Apply. **Clear** empties the
field through a real Kit GUI button before typing a replacement; tested remote
Ctrl+A and Home/End selection did not reliably replace Kit text. Cartesian
values use the existing world knife-edge pose `[x,y,z,qx,qy,qz,qw]`; Joint
values are the seven active-arm positions in radians. Cutting is separately
opted in by its checkbox. Invalid input is rejected without a physics step.

All GUI callbacks enqueue requests processed on the same Kit thread. Automatic
Kit timeline physics is disabled; the retained environment alone advances ten
native rigid substeps per control step. TLMPM uses measured articulation poses;
material, surface, RGB-D and environment timestamps are checked together.
Idle GUI rendering is verified not to advance physics. No task-control HTTP
endpoint, image replay or second robot simulator bypasses this loop.

## Stop and inspect

```bash
# Graceful shutdown: material/camera archives and final status are written.
touch artifacts/remote_session_NEW/STOP

# If started in the documented persistent sessions, stop HTTP separately.
tmux kill-session -t rcs-web-viewer
```

Foreground Ctrl+C also closes the live application. Preserve `events.jsonl`,
`status.json`, `result.json`, source hashes and every `episodes/` directory.
Each reset archives its transforms and material results; it does not overwrite
the previous episode. The HTTP service's `/health.json` checks the web service
only and explicitly does not assert simulation/video readiness.

## Actual validation and limits

The startup receipt in [REMOTE_VIEWER_RUNTIME_NOTES](REMOTE_VIEWER_RUNTIME_NOTES.md)
proves an actual Kit/PhysX/three-camera run with zero idle-step drift. Local
Chrome146 opened the actual Tailscale URL and decoded9191280×800 frames,
1,847,790 received video bytes and zero RTP packet loss. Nine streamed mouse
operations changed actual Kit state: camera selection, Step, Reset, Play,
Pause and task selection. Actual negotiated video was60fps;30fps was only the
client request. The selected media endpoint was100.111.162.17:47998/UDP.

Raw browser and application receipts remain in ignored
`artifacts/remote_web_viewer_browser_live_001` and
`artifacts/remote_interactive_001`. The no-server negative control and failed
keyboard replacement attempts remain preserved separately. Final keyboard
entry and external Mac acceptance are reported separately in the progress
receipt; local Chrome through a Tailscale address is not an independent Mac
connection test. Native D455 fidelity, ideal D405 limitations and task failures
are unchanged by the viewer.

The final Clear-button follow-up passed14 actual streamed GUI transactions,
including typed Cartesian and joint JSON targets followed by accepted actions
and coupled material clocks0.02s /0.04s. It decoded2,073frames /4,063,442video
bytes with zero packet loss; the actual input channel carried269messages /
7,679bytes. The browser disconnected after resetting TRIM to paused Global
at t=0, releasing the single-client slot. See [compact browser proof](evidence/remote_web_viewer.json)
and [actual full-GUI screenshot](media/remote_web_viewer.png). External Mac
verification remains pending until a real remote client connects.

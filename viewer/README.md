# Isaac Sim WebRTC viewer

This small client imports NVIDIA's official `@nvidia/ov-web-rtc` transport.
It displays the full live Kit GUI and forwards its mouse/keyboard input.
Task controls remain inside Kit; the web page has no task-command API.

From the repository root:

```bash
scripts/setup_remote_web_viewer.sh
.venv/bin/python scripts/serve_remote_web_viewer.py
```

Open `http://100.111.162.17:8210` in Chrome on the same Tailscale network.
The Isaac Sim interactive process must already be running. HTTP readiness is
separate from an established WebRTC/video connection.

See [deployment and actual validation](../docs/REMOTE_WEB_VIEWER.md).
[UPSTREAM.json](UPSTREAM.json) pins the official recipe, generator, SDK, Node
archive checksum and package lock. Only the original integration is tracked;
downloaded NVIDIA SDK code/licenses and compiled output stay in ignored local
dependencies. The installed SDK's own `LICENSE.txt` applies to that dependency.

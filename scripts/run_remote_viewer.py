#!/usr/bin/env python3
"""Stream the existing live Isaac Sim cutting runner with its full Kit GUI."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.remote_runtime import StreamingPorts, launch_streaming_app


def tailnet_ip():
    result = subprocess.run(["tailscale", "ip", "-4"], check=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("CUT-01", "TRIM-01"), default="TRIM-01")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ip", help="Host Tailscale IPv4; detected from tailscale ip -4 by default")
    parser.add_argument("--signal-port", type=int, default=49100)
    parser.add_argument("--media-port", type=int, default=47998)
    parser.add_argument("--web-port", type=int, default=8210)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max-wall-seconds", type=float, default=0., help="Optional bounded smoke;0 leaves the interactive session running")
    args = parser.parse_args()
    ip = args.ip or tailnet_ip()
    output = new_output(args.output_dir)
    config = IsaacSimCuttingConfig(task_id=args.task, gpu=args.gpu, device=f"cuda:{args.gpu}", seed=args.seed,
        camera_layout="wrist_d405_global_d455", camera_mode="native_single_view",
        particle_spacing_m=.002 if args.task == "TRIM-01" else None)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(output/"manifest.json", manifest({**asdict(config), "remote_viewer": arguments,
               "tailscale_ip": ip, "loop": "existing native PhysX + TLMPM; automatic GUI physics disabled"},
               "live_isaac_sim_remote_interactive"))
    implementation = output/"implementation"
    implementation.mkdir()
    for relative in ("scripts/run_remote_viewer.py", "src/robotic_cutting_sim/remote_viewer.py",
            "src/robotic_cutting_sim/remote_control.py", "src/robotic_cutting_sim/remote_runtime.py",
            "src/robotic_cutting_sim/isaac_sim_runner.py", "src/robotic_cutting_sim/live_env.py",
            "src/robotic_cutting_sim/camera_rig.py", "configs/isaac_sim_cutting_streaming.kit",
            "configs/wrist_camera_rig_v3.json"):
        shutil.copyfile(ROOT/relative, implementation/Path(relative).name)
    write_json(implementation/"sha256.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in implementation.iterdir() if p.is_file()})
    viewer_url = f"http://{ip}:{args.web_port}"
    write_json(output/"status.json", {"phase": "STARTING", "pid": os.getpid(), "viewer_url": viewer_url})
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    app, session = None, None
    ready = False
    try:
        app = launch_streaming_app(config, output, ip, args.width, args.height,
                                   StreamingPorts(args.signal_port, args.media_port))
        from robotic_cutting_sim.remote_viewer import RemoteCuttingSession
        session = RemoteCuttingSession(app, config, output)
        session.update()
        ready = True
        print(json.dumps({"status": "READY_CLIENT_CONNECTION_NOT_YET_VERIFIED", "viewer_url": viewer_url,
            "pid": os.getpid(), "output": str(output), "same_live_simulation": True}), flush=True)
        start = time.monotonic()
        while app.is_running() and not session.loop.closed and not stop[0] and not (output/"STOP").exists():
            if args.max_wall_seconds and time.monotonic()-start >= args.max_wall_seconds:
                break
            tick = time.monotonic()
            session.update()
            if not session.loop.playing:
                time.sleep(max(0., 1/30-(time.monotonic()-tick)))
        write_json(output/"result.json", {"status": "CLOSED", "initialized": ready,
            "same_live_application_pid": os.getpid(), "viewer_url": viewer_url,
            "last_state": session.loop.snapshot(), "full_task_acceptance": "SEPARATE_EVALUATION",
            "Mac_Chrome_client_acceptance": "REQUIRES_CLIENT_CONNECTION_EVIDENCE"})
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        detail = traceback.format_exc()
        (output/"failure.log").write_text(detail)
        write_json(output/"status.json", {"phase": "FAIL" if ready else "BLOCKED", "error": str(error),
            "traceback": detail, "pid": os.getpid(), "viewer_url": viewer_url,
            "standalone_or_replay_fallback": False})
        raise
    finally:
        if ready and session:
            existing = json.loads((output/"status.json").read_text())
            if existing.get("phase") == "READY":
                write_json(output/"status.json", {"phase": "CLOSED", "pid": os.getpid(),
                    "viewer_url": viewer_url, "last_state": session.loop.snapshot()})
        if session:
            try:
                session.close()
            except Exception:
                (output/"close_failure.log").write_text(traceback.format_exc())
        if app:
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())

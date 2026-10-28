#!/usr/bin/env python3
"""Serve only the official-WebRTC client; never start or control a simulator."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def port(value):
    value = int(value)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="100.111.162.17", help="HTTP bind address; defaults to this host's Tailscale IPv4")
    parser.add_argument("--port", type=port, default=8210)
    parser.add_argument("--signal-host", help="Browser-reachable Isaac Sim address; defaults to --host")
    parser.add_argument("--signal-port", type=port, default=49100)
    parser.add_argument("--media-port", type=port, default=47998)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    dist = ROOT/"viewer/dist"
    if not (dist/"index.html").is_file():
        parser.error("Viewer is not built. Run scripts/setup_remote_web_viewer.sh first.")
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("Width, height and FPS must be positive")
    config = {"signalingServer": args.signal_host or args.host, "signalingPort": args.signal_port,
        "mediaServer": args.signal_host or args.host, "mediaPort": args.media_port,
        "width": args.width, "height": args.height, "fps": args.fps}

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *values):
            # Record only public request metadata needed to distinguish an
            # actual remote browser from this host's automated check.
            print(json.dumps({"event": "http_request", "time": self.log_date_time_string(),
                "client_ip": self.client_address[0], "user_agent": self.headers.get("User-Agent", ""),
                "request": format % values}), flush=True)

        def do_GET(self):
            route = urlsplit(self.path).path
            payload = config if route == "/viewer-config.json" else {
                "service": "RoboticCuttingSim WebRTC viewer HTTP", "status": "HTTP_READY",
                "simulation_connection": "NOT_TESTED_BY_HTTP", "configuration": config
            } if route == "/health.json" else None
            if payload is None:
                return super().do_GET()
            data = (json.dumps(payload)+"\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(dist)))
    print(json.dumps({"viewer_url": f"http://{args.host}:{args.port}", "configuration": config,
        "scope": "HTTP client only; a live Isaac Sim WebRTC server is required"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

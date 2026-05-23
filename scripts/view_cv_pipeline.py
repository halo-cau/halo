"""Serve the interactive HALO CV pipeline viewer.

Usage:
    /home/ppco915/ENTER/envs/halo/bin/python scripts/view_cv_pipeline.py

Then open:
    http://127.0.0.1:8788/pipeline_viewer.html
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOM_DIR = PROJECT_ROOT / "server_room_phone"
DEFAULT_PORT = 8788


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with concise logs."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


class ReusableTCPServer(socketserver.TCPServer):
    """TCP server that can be restarted immediately on the same port."""

    allow_reuse_address = True


def serve_pipeline_viewer(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
    """Serve server_room_phone/ so the browser can load OBJ/PLY/JSON assets."""
    if not SERVER_ROOM_DIR.exists():
        raise FileNotFoundError(f"Missing viewer directory: {SERVER_ROOM_DIR}")

    handler = functools.partial(QuietHandler, directory=str(SERVER_ROOM_DIR))
    with ReusableTCPServer((host, port), handler) as httpd:
        display_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
        url = f"http://{display_host}:{port}/pipeline_viewer.html"
        print("HALO CV pipeline viewer")
        print(f"Serving: {SERVER_ROOM_DIR}")
        print(f"Open:    {url}")
        print("Controls: left-drag orbit, right-drag pan, scroll zoom")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped viewer server.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the interactive CV pipeline viewer.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port.")
    args = parser.parse_args()
    serve_pipeline_viewer(port=args.port, host=args.host)


if __name__ == "__main__":
    main()

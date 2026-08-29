#!/usr/bin/env python3
"""Zero-Compute Static File Server for NanoGuard.
All inference executes 100% client-side in the user's browser via WebGPU.
"""

from __future__ import annotations

import http.server
from pathlib import Path
import socket
import sys

WEB_DIR = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class NanoGuardHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler delivering HTML, JS, CSS, and ONNX model assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()


class DualStackThreadingServer(http.server.ThreadingHTTPServer):
    """Dual-stack HTTP server supporting both IPv4 (127.0.0.1) and IPv6 (localhost)."""
    address_family = socket.AF_INET6

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def main():
    try:
        httpd = DualStackThreadingServer(("", PORT), NanoGuardHandler)
    except Exception as exc:
        print(f"[Server] Dual-stack fallback to IPv4: {exc}")
        httpd = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), NanoGuardHandler)

    httpd.allow_reuse_address = True
    print(f"\n========================================================")
    print(f"  NanoGuard Static Server (Pure Client-Side WebGPU)")
    print(f"  URL: http://localhost:{PORT}")
    print(f"  Server ML Compute: 0% (Zero server inference)")
    print(f"  Execution Target: Client-Side WebGPU")
    print(f"========================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")


if __name__ == "__main__":
    main()

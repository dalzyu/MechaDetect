#!/usr/bin/env python3
"""Static file server for the MechaDetect client-side WebGPU demo.

The server never performs inference. It only serves the frontend and ONNX
model, with optional TLS for browsers that require a secure context for
WebGPU (notably iPhone Safari).
"""

from __future__ import annotations

import argparse
import http.server
from pathlib import Path
import socket
import ssl

WEB_DIR = Path(__file__).resolve().parent


class MechaDetectHandler(http.server.SimpleHTTPRequestHandler):
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
    """Dual-stack HTTP server supporting both IPv4 and IPv6 when available."""

    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the MechaDetect WebGPU demo to local or remote devices."
    )
    parser.add_argument(
        "legacy_port",
        nargs="?",
        type=int,
        help="legacy positional port (prefer --port)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="interface to bind (default: 0.0.0.0, all IPv4 interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port (default: 8000)",
    )
    parser.add_argument(
        "--certfile",
        type=Path,
        help="PEM certificate chain; enables HTTPS when paired with --keyfile",
    )
    parser.add_argument(
        "--keyfile",
        type=Path,
        help="PEM private key paired with --certfile",
    )
    args = parser.parse_args()
    if args.certfile is not None and args.keyfile is None:
        parser.error("--certfile requires --keyfile")
    if args.keyfile is not None and args.certfile is None:
        parser.error("--keyfile requires --certfile")
    if args.port is not None and args.legacy_port is not None:
        parser.error("provide either the positional port or --port, not both")
    args.port = args.port if args.port is not None else (args.legacy_port or 8000)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def create_server(host: str, port: int) -> http.server.ThreadingHTTPServer:
    """Create the best available server for the requested bind address."""

    try:
        if ":" in host or host in ("", "localhost"):
            return DualStackThreadingServer((host, port), MechaDetectHandler)
    except OSError as exc:
        print(f"[Server] Dual-stack fallback to IPv4: {exc}")
    return http.server.ThreadingHTTPServer(
        (host or "0.0.0.0", port),
        MechaDetectHandler,
    )


def enable_tls(
    httpd: http.server.ThreadingHTTPServer,
    certfile: Path,
    keyfile: Path,
) -> None:
    """Wrap an already-bound server socket in a TLS context."""

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)


def main() -> None:
    args = parse_args()
    httpd = create_server(args.host, args.port)
    scheme = "http"
    if args.certfile is not None and args.keyfile is not None:
        enable_tls(httpd, args.certfile, args.keyfile)
        scheme = "https"

    httpd.allow_reuse_address = True
    display_host = args.host if args.host not in ("", "0.0.0.0", "::") else "<this-PC-LAN-IP>"
    print("\n========================================================")
    print("  MechaDetect Static Server (Client-Side WebGPU)")
    print(f"  Local URL: http://127.0.0.1:{args.port}")
    print(f"  Device URL: {scheme}://{display_host}:{args.port}")
    print("  Server ML Compute: 0% (Zero server inference)")
    print("  Execution Target: Client-Side WebGPU")
    if scheme == "http":
        print("  Note: iPhone Safari requires HTTPS for WebGPU outside localhost.")
    print("========================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

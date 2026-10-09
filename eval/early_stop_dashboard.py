#!/usr/bin/env python3
"""Serve the local live-training dashboard from a run's evaluation folder."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="checkpoints/<name>/evaluation_dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Folder dashboard belum ada: {root}. Mulai training dengan evaluasi aktif dulu.")
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print("Tekan Ctrl+C untuk menutup server dashboard.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard dihentikan.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

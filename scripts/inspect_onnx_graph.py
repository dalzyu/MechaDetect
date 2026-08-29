#!/usr/bin/env python3
"""Inspect ONNX models for static INT8 verification and reject INT4/dynamic claims."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
_onnx_pkg = REPO_ROOT / ".runtime" / "onnx_packages"
if _onnx_pkg.exists() and str(_onnx_pkg) not in sys.path:
    sys.path.insert(0, str(_onnx_pkg))

from scripts.quantize_webgpu_nbits import inspect_model_cli


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect ONNX models for static INT8 verification and reject INT4/dynamic claims."
    )
    parser.add_argument(
        "--inspect-model",
        type=Path,
        required=True,
        help="Inspect an ONNX model to verify static INT8 and reject INT4/dynamic-only claims",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional destination path for machine-readable inspection JSON",
    )
    args = parser.parse_args()

    exit_code = inspect_model_cli(args.inspect_model, args.report_json)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

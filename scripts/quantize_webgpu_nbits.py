#!/usr/bin/env python3
"""Inspect ONNX models for static INT8 verification and reject INT4/dynamic claims.

Provides machine-verifiable graph inspection to prove static INT8 quantization and
strictly reject INT4 MatMulNBits or dynamic-only claims.
"""

from __future__ import annotations

import argparse
import json
import logging
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

import onnx

from aigc_detector.static_int8 import inspect_and_verify_static_int8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def inspect_model_cli(model_path: Path, output_json: Path | None = None) -> int:
    """Inspect ONNX model graph to verify static INT8 and reject INT4 or dynamic-only claims."""
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        return 2

    logger.info("Inspecting ONNX graph structure for: %s", model_path)
    result = inspect_and_verify_static_int8(model_path)
    result_dict = result.to_dict()

    print("\n=======================================================")
    print("           ONNX GRAPH VERIFICATION REPORT")
    print("=======================================================")
    print(f"Model Path:              {model_path}")
    print(f"File Size:               {model_path.stat().st_size / 1e6:.2f} MB")
    print(f"Total Graph Nodes:       {result.total_nodes}")
    print(f"Quantization Type:       {result.quantization_type}")
    print(f"Static INT8 Verified:    {result.static_int8_verified}")
    print(f"INT4 Detected:           {result.int4_detected} (MatMulNBits={result.matmul_nbits_count})")
    print(f"Dynamic-Only Detected:   {result.dynamic_only_detected} (DynamicQuant={result.dynamic_quant_count})")
    print(f"QDQ Node Count:          {result.qdq_node_count}")
    print(f"QLinear Node Count:      {result.qlinear_node_count}")
    print(f"Preserved Sensitive Ops: {result.preserved_sensitive_ops}")
    print("-------------------------------------------------------")
    print(f"VERIFICATION STATUS:     {'PASSED (Static INT8)' if result.passed else 'FAILED'}")

    if result.rejection_reasons:
        print("\nRejection Reasons:")
        for r in result.rejection_reasons:
            print(f"  ❌ {r}")
    else:
        print("\n  ✅ Verified static INT8 QDQ graph. Zero MatMulNBits. Zero dynamic-only quant.")
    print("=======================================================\n")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        logger.info("Verification JSON report saved to: %s", output_json)

    return 0 if result.passed else 1


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

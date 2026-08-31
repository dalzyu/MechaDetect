#!/usr/bin/env python3
"""Calibrate and quantize student models to static INT8 for WebGPU/WASM deployment.

Enforces the 4096-row disjoint calibration dataset contract, guards against canonical
split overlap, requires authoritative float metadata sidecars to inherit calibrated threshold
and manifest digest, quantizes supported conv/linear operators while preserving sensitive ops,
inspects the graph to verify static INT8 and reject INT4 MatMulNBits or dynamic-only claims,
marks un-evaluated PTQ as experimental, and emits verified post-embed SHA-256 sidecars.
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

from typing import Any

import onnx
from onnxruntime.quantization import (
    QuantFormat,
    QuantType,
    quantize_static,
)

from aigc_detector.runtime import load_local_environment
from aigc_detector.static_int8 import (
    DisjointCalibrationDataReader,
    build_artifact_metadata,
    evaluate_onnx_float_vs_int8,
    finalize_artifact_and_sidecar,
    get_nodes_to_exclude_from_quantization,
    inspect_and_verify_static_int8,
    load_manifest_records,
    normalize_qdq_parameter_ranks_for_webgpu,
    verify_calibration_disjointness,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_metadata_from_onnx(model: onnx.ModelProto) -> dict[str, Any]:
    """Extract embedded metadata properties from an ONNX model."""
    meta: dict[str, Any] = {}
    for prop in model.metadata_props:
        try:
            meta[prop.key] = json.loads(prop.value)
        except (json.JSONDecodeError, TypeError):
            meta[prop.key] = prop.value
    return meta


def resolve_float_metadata(
    input_model_path: Path,
    float_metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve and validate authoritative float model metadata sidecar.

    Inherits calibrated_threshold and manifest_digest; fails closed if absent.
    """
    if float_metadata_path is not None and Path(float_metadata_path).is_file():
        logger.info("Loading float model metadata from: %s", float_metadata_path)
        with open(float_metadata_path, encoding="utf-8") as f:
            return json.load(f)

    # Check companion metadata files alongside input model
    stem = input_model_path.stem
    candidates = [
        input_model_path.with_name(f"{stem}.metadata.json"),
        input_model_path.parent / f"{stem.replace('_float', '')}.metadata.json",
        input_model_path.parent / "metadata.json",
    ]
    for cand in candidates:
        if cand.is_file():
            logger.info("Auto-discovered float metadata sidecar at: %s", cand)
            with open(cand, encoding="utf-8") as f:
                return json.load(f)

    # Fallback to embedded metadata in float ONNX model
    float_proto = onnx.load(str(input_model_path))
    embedded = extract_metadata_from_onnx(float_proto)
    if embedded and embedded.get("manifest_digest"):
        logger.info("Extracted authoritative metadata embedded in input ONNX model.")
        return embedded

    raise FileNotFoundError(
        f"Authoritative float metadata sidecar not found for '{input_model_path}'. "
        "Static INT8 PTQ strictly requires float metadata to inherit calibrated_threshold, "
        "manifest_digest, and model identity. Please specify --float-metadata."
    )


def calibrate_and_quantize_int8(
    input_model_path: Path,
    calibration_manifest_path: Path,
    canonical_manifests_dir: Path | None = None,
    data_root: Path | None = None,
    output_model_path: Path | None = None,
    num_calibration_samples: int = 4096,
    allow_non_4096: bool = False,
    float_metadata_path: Path | None = None,
    metadata_output: Path | None = None,
    eval_manifest_path: Path | None = None,
    force_experimental: bool = False,
    tolerance_auroc_drop: float = 0.005,  # 0.5 percentage points
) -> tuple[Path, Path, dict[str, Any]]:
    """Execute static INT8 calibration, quantization, graph verification, and metadata emission."""
    load_local_environment(REPO_ROOT)
    input_model_path = Path(input_model_path).resolve()
    calibration_manifest_path = Path(calibration_manifest_path).resolve()

    if not input_model_path.exists():
        raise FileNotFoundError(f"Input float ONNX model not found: {input_model_path}")

    # 1. Require and load authoritative float metadata sidecar
    float_meta = resolve_float_metadata(input_model_path, float_metadata_path)

    calibrated_threshold = float(float_meta["calibrated_threshold"])
    manifest_digest = str(float_meta["manifest_digest"])
    model_family = str(float_meta.get("model_family", "dinov3_student"))
    variant = str(float_meta.get("variant", "small"))
    stage = str(float_meta.get("stage", "post_att"))
    param_count = int(float_meta.get("parameter_count", 0))
    float_status = str(float_meta.get("evaluation_status", "experimental"))
    preprocessing_version = str(float_meta.get("preprocessing_version", "2"))
    opset_version = int(float_meta.get("opset_version", 17))
    logger.info(
        "Inherited float metadata: threshold=%.4f, manifest_digest=%s, variant=%s",
        calibrated_threshold,
        manifest_digest[:16],
        variant,
    )

    # 2. Load calibration manifest
    logger.info("Loading calibration manifest from: %s", calibration_manifest_path)
    calib_records = load_manifest_records(calibration_manifest_path)
    logger.info("Loaded %d calibration records.", len(calib_records))

    # 3. Strict disjointness verification against canonical splits
    logger.info("Verifying calibration dataset disjointness against canonical splits...")
    disjoint_report = verify_calibration_disjointness(
        calibration_records=calib_records,
        canonical_manifests_dir=canonical_manifests_dir,
        allow_non_4096=allow_non_4096,
    )
    if not disjoint_report.passed:
        error_msg = "Disjointness guard failed! Calibration data rejected:\n" + "\n".join(
            f" - {r}" for r in disjoint_report.rejection_reasons
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        "Disjointness guard PASSED: %d calibration rows strictly disjoint from %d canonical rows.",
        disjoint_report.calibration_row_count,
        disjoint_report.canonical_row_count,
    )

    # 4. Resolve output path
    if output_model_path is None:
        out_dir = REPO_ROOT / "outputs" / "exported_models"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_model_path = out_dir / f"student_{variant}_{stage}_static_int8.onnx"
    else:
        output_model_path = Path(output_model_path).resolve()
        output_model_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Identify nodes to exclude (preserve sensitive ops)
    float_proto = onnx.load(str(input_model_path))
    nodes_to_exclude = get_nodes_to_exclude_from_quantization(float_proto)
    logger.info(
        "Preserving %d sensitive nodes in float32 (Softmax, Sigmoid, LayerNorm, dynamic attention matmuls, classifier)",
        len(nodes_to_exclude),
    )

    # 6. Build CalibrationDataReader
    reader = DisjointCalibrationDataReader(
        records=calib_records,
        data_root=data_root,
        batch_size=1,
        max_samples=num_calibration_samples,
    )

    # 7. Run static INT8 quantization
    logger.info("Executing static INT8 QDQ quantization to: %s...", output_model_path)
    quantize_static(
        model_input=str(input_model_path),
        model_output=str(output_model_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        nodes_to_exclude=nodes_to_exclude,
    )
    webgpu_rank_fixes = normalize_qdq_parameter_ranks_for_webgpu(output_model_path)
    logger.info(
        "Normalized %d QDQ parameter-rank mismatch(es) for ONNX Runtime WebGPU.",
        len(webgpu_rank_fixes),
    )

    # 8. Inspect graph to prove static INT8 and reject INT4 / dynamic-only claims
    logger.info("Inspecting quantized ONNX graph structure...")
    graph_report = inspect_and_verify_static_int8(output_model_path)

    if not graph_report.passed:
        reasons = "\n".join(f" - {r}" for r in graph_report.rejection_reasons)
        error_msg = f"Quantization verification FAILED:\n{reasons}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(
        "Static INT8 graph verification PASSED: QDQ nodes=%d, MatMulNBits=%d (zero=verified), sensitive ops preserved=%s",
        graph_report.qdq_node_count,
        graph_report.matmul_nbits_count,
        graph_report.preserved_sensitive_ops,
    )

    # 9. Verify file size reduction (smallest artifact contract)
    float_size = input_model_path.stat().st_size
    int8_size = output_model_path.stat().st_size
    size_ratio = int8_size / float_size
    logger.info(
        "Size comparison: Float=%.2f MB, Static INT8=%.2f MB (ratio=%.2f, compression=%.1f%%)",
        float_size / 1e6,
        int8_size / 1e6,
        size_ratio,
        (1.0 - size_ratio) * 100.0,
    )

    # 10. Evaluate float vs INT8 on evaluation manifest
    eval_results: dict[str, Any] = {}
    gate_passed = False
    evaluation_status = "experimental"

    if eval_manifest_path is not None and Path(eval_manifest_path).exists():
        logger.info(
            "Evaluating float vs INT8 parity on evaluation manifest: %s", eval_manifest_path
        )
        eval_records = load_manifest_records(Path(eval_manifest_path))
        eval_results = evaluate_onnx_float_vs_int8(
            float_model_path=input_model_path,
            int8_model_path=output_model_path,
            eval_records=eval_records,
            data_root=data_root,
            calibrated_threshold=calibrated_threshold,
            tolerance_auroc_drop=tolerance_auroc_drop,
        )
        gate_passed = bool(eval_results.get("passed_gate", False))
        if gate_passed and not force_experimental and float_status == "promoted":
            evaluation_status = "promoted"
            logger.info(
                "Static INT8 artifact PASSED quality gate (AUROC drop <= %.4f). Status set to 'promoted'.",
                tolerance_auroc_drop,
            )
        else:
            evaluation_status = "experimental"
            logger.warning(
                "Static INT8 artifact marked as 'experimental' (gate_passed=%s, force_experimental=%s, float_status='%s'). "
                "The promoted float/ATT student model remains default.",
                gate_passed,
                force_experimental,
                float_status,
            )
    else:
        # Missing evaluation manifest: must be experimental, NEVER promoted
        gate_passed = False
        evaluation_status = "experimental"
        logger.warning(
            "No evaluation manifest provided: static INT8 model marked as 'experimental'. "
            "Never promoted without verified evaluation. The promoted float/ATT student model remains default."
        )

    # 11. Build artifact metadata contract
    metadata = build_artifact_metadata(
        model_family=model_family,
        variant=variant,
        stage=stage,
        parameter_count=param_count,
        quantization="static_int8",
        calibrated_threshold=calibrated_threshold,
        input_size=[3, 224, 224],
        preprocessing_version=preprocessing_version,
        manifest_digest=manifest_digest,
        evaluation_status=evaluation_status,
        opset_version=opset_version,
        checkpoint_path=float_meta.get("checkpoint_path", ""),
        checkpoint_sha256=float_meta.get("checkpoint_sha256", ""),
        artifact_path=str(output_model_path),
        parity_verified=gate_passed,
        extra={
            "float_artifact_path": str(input_model_path),
            "float_artifact_size_bytes": float_size,
            "compression_ratio": float(size_ratio),
            "graph_verification": graph_report.to_dict(),
            "disjointness_verification": disjoint_report.to_dict(),
            "evaluation_metrics": eval_results,
            "gate_passed": gate_passed,
        },
    )

    # 12. Embed metadata into INT8 ONNX, compute verified post-embed SHA-256 and size, write sidecar
    final_int8_sha, final_int8_size, meta_path = finalize_artifact_and_sidecar(
        onnx_path=output_model_path,
        metadata=metadata,
        sidecar_path=metadata_output,
    )
    logger.info(
        "Artifact metadata written to: %s (final sha256=%s...)", meta_path, final_int8_sha[:16]
    )

    return output_model_path, meta_path, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and quantize student model to static INT8 for WebGPU/WASM."
    )
    parser.add_argument(
        "--input-model",
        type=Path,
        required=True,
        help="Path to exported float ONNX model",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        required=True,
        help="Path to 4096-row calibration manifest (.parquet or .jsonl)",
    )
    parser.add_argument(
        "--canonical-manifests-dir",
        type=Path,
        default=None,
        help="Directory with canonical split manifests for disjointness guard",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root directory containing image files",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help="Path for static INT8 ONNX output (defaults to collision-proof naming)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=4096,
        help="Number of calibration samples (contract: exactly 4096)",
    )
    parser.add_argument(
        "--allow-non-4096",
        action="store_true",
        help="Permit non-4096 row counts for testing or debugging",
    )
    parser.add_argument(
        "--float-metadata",
        type=Path,
        default=None,
        help="Path to authoritative float model metadata sidecar JSON",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Destination for companion metadata JSON",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Optional evaluation manifest to assess float vs INT8 degradation",
    )
    parser.add_argument(
        "--force-experimental",
        action="store_true",
        help="Force artifact evaluation_status to 'experimental'",
    )
    parser.add_argument(
        "--tolerance-auroc-drop",
        type=float,
        default=0.005,
        help="Maximum permitted AUROC degradation for promotion (default: 0.005 = 0.5%%)",
    )
    args = parser.parse_args()

    calibrate_and_quantize_int8(
        input_model_path=args.input_model,
        calibration_manifest_path=args.calibration_manifest,
        canonical_manifests_dir=args.canonical_manifests_dir,
        data_root=args.data_root,
        output_model_path=args.output_model,
        num_calibration_samples=args.num_calibration_samples,
        allow_non_4096=args.allow_non_4096,
        float_metadata_path=args.float_metadata,
        metadata_output=args.metadata_output,
        eval_manifest_path=args.eval_manifest,
        force_experimental=args.force_experimental,
        tolerance_auroc_drop=args.tolerance_auroc_drop,
    )


if __name__ == "__main__":
    main()

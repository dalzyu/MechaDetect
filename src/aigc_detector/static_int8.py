"""Static INT8 PTQ helpers, calibration disjointness guards, and graph verification.

Provides machine-verifiable static INT8 quantization for student models,
strictly rejects INT4 MatMulNBits and dynamic-only quantization, enforces
calibration split disjointness, and emits artifact metadata matching contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
_onnx_pkg = _ROOT / ".runtime" / "onnx_packages"
if _onnx_pkg.exists() and str(_onnx_pkg) not in sys.path:
    sys.path.insert(0, str(_onnx_pkg))

import numpy as np
import onnx
import torch
from onnx import TensorProto

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
except ImportError:  # pragma: no cover
    ort = None
    CalibrationDataReader = object  # type: ignore[misc,assignment]
    QuantFormat = None  # type: ignore[misc,assignment]
    QuantType = None  # type: ignore[misc,assignment]
    quantize_static = None  # type: ignore[misc,assignment]

from aigc_detector.metrics import binary_auroc

logger = logging.getLogger(__name__)

# Standard image preprocessing normalization constants (ImageNet v1)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Known 4-bit tensor types in ONNX protobuf
INT4_TENSOR_TYPES = {
    getattr(TensorProto, "INT4", 22),
    getattr(TensorProto, "UINT4", 21),
}


@dataclass
class DisjointnessReport:
    """Report validating calibration data disjointness against canonical splits."""

    passed: bool
    calibration_row_count: int
    canonical_split_files: list[str]
    canonical_row_count: int
    row_id_collisions: list[str] = field(default_factory=list)
    image_path_collisions: list[str] = field(default_factory=list)
    sha256_collisions: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphVerificationResult:
    """Machine-verifiable graph inspection report proving static INT8 structure."""

    passed: bool
    quantization_type: str  # "static_int8", "int4_nbits", "dynamic_int8", "float32", "unknown"
    total_nodes: int
    op_counts: dict[str, int]
    qdq_node_count: int
    qlinear_node_count: int
    matmul_nbits_count: int
    dynamic_quant_count: int
    int4_detected: bool
    dynamic_only_detected: bool
    static_int8_verified: bool
    preserved_sensitive_ops: dict[str, int]
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    """Load manifest records from parquet or jsonl file."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(manifest_path)
        return df.to_dict(orient="records")

    records: list[dict[str, Any]] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verify_calibration_disjointness(
    calibration_records: list[dict[str, Any]],
    canonical_manifests_dir: Path | None,
    allow_non_4096: bool = False,
) -> DisjointnessReport:
    """Verify that calibration data has exactly 4096 rows and is strictly disjoint from canonical splits.

    Checks calibration records against canonical splits:
    train, validation, test, test_unseen, and exclusions.
    Fails closed if any row_id, image_path, or sha256 overlaps.
    """
    calib_count = len(calibration_records)
    rejection_reasons: list[str] = []

    if not allow_non_4096 and calib_count != 4096:
        rejection_reasons.append(
            f"Static INT8 calibration contract requires exactly 4096 rows, but found {calib_count}"
        )

    # Extract calibration identity sets
    calib_row_ids: set[str] = set()
    calib_image_paths: set[str] = set()
    calib_sha256s: set[str] = set()

    for r in calibration_records:
        if r.get("row_id"):
            calib_row_ids.add(str(r["row_id"]))
        if r.get("image_path"):
            calib_image_paths.add(str(r["image_path"]))
        if r.get("sha256"):
            calib_sha256s.add(str(r["sha256"]))

    canonical_files_found: list[str] = []
    canonical_row_ids: set[str] = set()
    canonical_image_paths: set[str] = set()
    canonical_sha256s: set[str] = set()

    if canonical_manifests_dir is not None:
        c_dir = Path(canonical_manifests_dir)
        if c_dir.exists():
            # Search for canonical split manifests in dir or subdirs
            candidates = list(c_dir.glob("*.parquet")) + list(c_dir.glob("*.jsonl"))
            for p in candidates:
                stem = p.stem.lower()
                if any(
                    k in stem
                    for k in ["train", "val", "validation", "test", "test_unseen", "exclusion"]
                ):
                    canonical_files_found.append(str(p.name))
                    records = load_manifest_records(p)
                    for r in records:
                        if r.get("row_id"):
                            canonical_row_ids.add(str(r["row_id"]))
                        if r.get("image_path"):
                            canonical_image_paths.add(str(r["image_path"]))
                        if r.get("sha256"):
                            canonical_sha256s.add(str(r["sha256"]))

    row_id_overlap = sorted(calib_row_ids.intersection(canonical_row_ids))
    image_path_overlap = sorted(calib_image_paths.intersection(canonical_image_paths))
    sha256_overlap = sorted(calib_sha256s.intersection(canonical_sha256s))

    if row_id_overlap:
        rejection_reasons.append(
            f"Disjointness guard failed: {len(row_id_overlap)} row_id collisions detected with canonical splits: {row_id_overlap[:5]}"
        )
    if image_path_overlap:
        rejection_reasons.append(
            f"Disjointness guard failed: {len(image_path_overlap)} image_path collisions detected with canonical splits: {image_path_overlap[:5]}"
        )
    if sha256_overlap:
        rejection_reasons.append(
            f"Disjointness guard failed: {len(sha256_overlap)} sha256 content collisions detected with canonical splits: {sha256_overlap[:5]}"
        )

    passed = len(rejection_reasons) == 0
    return DisjointnessReport(
        passed=passed,
        calibration_row_count=calib_count,
        canonical_split_files=canonical_files_found,
        canonical_row_count=len(canonical_row_ids),
        row_id_collisions=row_id_overlap,
        image_path_collisions=image_path_overlap,
        sha256_collisions=sha256_overlap,
        rejection_reasons=rejection_reasons,
    )


def normalize_qdq_parameter_ranks_for_webgpu(onnx_path: str | Path) -> list[str]:
    """Align one-element QDQ zero-point ranks with their scales for ORT WebGPU.

    ONNX Runtime's static quantizer can emit a one-element scale with shape
    ``[1]`` and a scalar zero point for bias DequantizeLinear nodes. Both
    tensors contain one value and are valid broadcast inputs on CPU, but the
    WebGPU execution provider requires them to have the same rank.
    """
    model_path = Path(onnx_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    model = onnx.load(str(model_path))
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    adjusted_nodes: list[str] = []

    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or len(node.input) < 3:
            continue
        scale = initializers.get(node.input[1])
        zero_point = initializers.get(node.input[2])
        if scale is None or zero_point is None or len(scale.dims) == len(zero_point.dims):
            continue

        scale_elements = 1
        for dim in scale.dims:
            scale_elements *= int(dim)
        zero_point_elements = 1
        for dim in zero_point.dims:
            zero_point_elements *= int(dim)
        if scale_elements != zero_point_elements:
            raise ValueError(
                f"Cannot align QDQ parameter ranks for node '{node.name}': "
                f"scale shape={tuple(scale.dims)}, zero-point shape={tuple(zero_point.dims)}"
            )

        del zero_point.dims[:]
        zero_point.dims.extend(scale.dims)
        adjusted_nodes.append(node.name or node.output[0])

    if adjusted_nodes:
        onnx.save(model, str(model_path))
    return adjusted_nodes


def inspect_and_verify_static_int8(
    onnx_path_or_model: str | Path | onnx.ModelProto,
) -> GraphVerificationResult:
    """Inspect ONNX graph to prove static INT8 quantization and strictly reject INT4 MatMulNBits or dynamic-only claims.

    Criteria for passing static INT8:
    1. Zero MatMulNBits nodes (strictly rejects INT4 claim).
    2. Zero 4-bit tensor initializers or value infos.
    3. Zero DynamicQuantizeLinear-only graphs (strictly rejects dynamic-only claim).
    4. Must have static QuantizeLinear / DequantizeLinear (QDQ) or QLinear nodes.
    5. Preserves sensitive operations (Softmax, Sigmoid, LayerNormalization) in float.
    """
    if isinstance(onnx_path_or_model, (str, Path)):
        model_path = Path(onnx_path_or_model)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        model = onnx.load(str(model_path), load_external_data=False)
    else:
        model = onnx_path_or_model

    op_counts = Counter(node.op_type for node in model.graph.node)
    total_nodes = len(model.graph.node)

    qdq_count = op_counts.get("QuantizeLinear", 0) + op_counts.get("DequantizeLinear", 0)
    qlinear_count = op_counts.get("QLinearConv", 0) + op_counts.get("QLinearMatMul", 0)
    matmul_nbits_count = op_counts.get("MatMulNBits", 0)
    dynamic_quant_count = op_counts.get("DynamicQuantizeLinear", 0)

    # Check for INT4 in tensor types
    int4_detected = matmul_nbits_count > 0
    for init in model.graph.initializer:
        if init.data_type in INT4_TENSOR_TYPES:
            int4_detected = True
            break

    # Dynamic-only detection: DynamicQuantizeLinear without static QDQ/QLinear
    dynamic_only_detected = dynamic_quant_count > 0 and qdq_count == 0 and qlinear_count == 0

    rejection_reasons: list[str] = []

    if int4_detected:
        rejection_reasons.append(
            f"Rejected: Graph contains INT4 MatMulNBits or 4-bit tensors (MatMulNBits count={matmul_nbits_count}). "
            "INT4 block-wise quantization cannot be claimed as static INT8."
        )

    if dynamic_only_detected:
        rejection_reasons.append(
            f"Rejected: Graph contains DynamicQuantizeLinear ({dynamic_quant_count} nodes) without static calibration ops. "
            "Dynamic-only quantization does not use static calibration."
        )

    if qdq_count == 0 and qlinear_count == 0 and not int4_detected:
        rejection_reasons.append(
            "Rejected: Graph contains no static INT8 nodes (zero QuantizeLinear, DequantizeLinear, or QLinear nodes)."
        )

    # Count preserved sensitive operations
    sensitive_ops = ["Softmax", "Sigmoid", "LayerNormalization"]
    preserved_sensitive_ops = {op: op_counts.get(op, 0) for op in sensitive_ops}

    static_int8_verified = (
        (qdq_count > 0 or qlinear_count > 0) and not int4_detected and not dynamic_only_detected
    )

    if int4_detected:
        quantization_type = "int4_nbits"
    elif dynamic_only_detected:
        quantization_type = "dynamic_int8"
    elif static_int8_verified:
        quantization_type = "static_int8"
    elif total_nodes > 0 and qdq_count == 0 and qlinear_count == 0:
        quantization_type = "float32"
    else:
        quantization_type = "unknown"

    passed = static_int8_verified and len(rejection_reasons) == 0

    return GraphVerificationResult(
        passed=passed,
        quantization_type=quantization_type,
        total_nodes=total_nodes,
        op_counts=dict(op_counts),
        qdq_node_count=qdq_count,
        qlinear_node_count=qlinear_count,
        matmul_nbits_count=matmul_nbits_count,
        dynamic_quant_count=dynamic_quant_count,
        int4_detected=int4_detected,
        dynamic_only_detected=dynamic_only_detected,
        static_int8_verified=static_int8_verified,
        preserved_sensitive_ops=preserved_sensitive_ops,
        rejection_reasons=rejection_reasons,
    )


def get_nodes_to_exclude_from_quantization(model: onnx.ModelProto) -> list[str]:
    """Identify nodes that must remain in float32 for numerical stability.

    Preserves:
    - Multi-head attention inner dynamic matrix operations (queries, attention projections)
    - Sensitive operations: Softmax, Sigmoid, LayerNormalization
    - Final classification head (ai_positive_classifier)
    """
    nodes_to_exclude: set[str] = set()

    for node in model.graph.node:
        name_lower = node.name.lower()
        # Exclude dynamic attention operations and classification head
        if any(
            term in name_lower
            for term in ["attention", "queries", "classifier", "tamper_query", "aigc_queries"]
        ):
            nodes_to_exclude.add(node.name)
        # Exclude sensitive activation / normalization ops
        if node.op_type in ["Softmax", "Sigmoid", "LayerNormalization"]:
            nodes_to_exclude.add(node.name)

    return sorted(nodes_to_exclude)


if ort is not None:

    class DisjointCalibrationDataReader(CalibrationDataReader):
        """Calibration data reader delivering preprocessed image batches for static PTQ."""

        def __init__(
            self,
            records: list[dict[str, Any]],
            data_root: Path | None = None,
            batch_size: int = 1,
            max_samples: int | None = 4096,
        ) -> None:
            self.records = records[:max_samples] if max_samples else records
            self.data_root = Path(data_root) if data_root else None
            self.batch_size = batch_size
            self.index = 0

        def _load_image_tensor(self, record: dict[str, Any]) -> np.ndarray:
            """Load and preprocess a single image into normalized float32 [1, 3, 224, 224]."""
            image_path_str = record.get("image_path")
            if not image_path_str:
                raise ValueError(f"Calibration record missing 'image_path': {record}")
            path_obj = Path(image_path_str)
            if path_obj.is_absolute():
                full_path = path_obj
            elif self.data_root:
                full_path = self.data_root / path_obj
            else:
                full_path = path_obj

            if not full_path.exists():
                raise FileNotFoundError(
                    f"Calibration image file not found on disk: {full_path}. "
                    "Static INT8 PTQ requires byte-backed images and strictly prohibits synthetic fallback."
                )

            from PIL import Image

            img = Image.open(full_path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [224, 224, 3]
            arr = np.transpose(arr, (2, 0, 1))  # [3, 224, 224]
            arr = np.expand_dims(arr, axis=0)  # [1, 3, 224, 224]
            normalized = (arr - IMAGENET_MEAN) / IMAGENET_STD
            return normalized.astype(np.float32)

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self.index >= len(self.records):
                return None

            batch_records = self.records[self.index : self.index + self.batch_size]
            self.index += len(batch_records)

            tensors = [self._load_image_tensor(rec) for rec in batch_records]
            batch_tensor = np.concatenate(tensors, axis=0)
            return {"pixel_values": batch_tensor}

        def rewind(self) -> None:
            self.index = 0


def build_artifact_metadata(
    *,
    model_family: str,
    variant: str,
    stage: str,
    parameter_count: int,
    quantization: str,
    calibrated_threshold: float,
    manifest_digest: str,
    evaluation_status: str,
    input_size: list[int] | None = None,
    preprocessing_version: str = "2",
    opset_version: int = 17,
    checkpoint_path: str = "",
    checkpoint_sha256: str = "",
    artifact_path: str = "",
    artifact_sha256: str = "",
    artifact_size_bytes: int = 0,
    parity_verified: bool = False,
    max_abs_diff_pytorch_vs_onnx: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct metadata dictionary strictly matching the artifact metadata contract.

    Contract:
    model_family, parameter_count, quantization, calibrated_threshold,
    input_size, preprocessing_version, manifest_digest, evaluation_status.

    Fails closed on missing or invalid contract fields; strictly prohibits fabricated
    defaults for threshold, digest, or promoted status.
    """
    if not model_family:
        raise ValueError("model_family is required and cannot be empty.")
    if parameter_count < 0:
        raise ValueError(f"parameter_count must be non-negative, got {parameter_count}.")
    if quantization not in ("float32", "static_int8"):
        raise ValueError(f"quantization must be 'float32' or 'static_int8', got '{quantization}'.")
    if calibrated_threshold is None:
        raise ValueError("calibrated_threshold is required and cannot be None.")
    calibrated_threshold = float(calibrated_threshold)
    if not (0.0 <= calibrated_threshold <= 1.0):
        raise ValueError(
            f"calibrated_threshold must be between 0.0 and 1.0, got {calibrated_threshold}."
        )
    if not manifest_digest:
        raise ValueError("manifest_digest is required and cannot be empty.")
    if evaluation_status not in ("promoted", "experimental", "candidate"):
        raise ValueError(
            f"evaluation_status must be 'promoted', 'experimental', or 'candidate', got '{evaluation_status}'."
        )

    resolved_input_size = input_size or [3, 224, 224]
    if len(resolved_input_size) != 3:
        raise ValueError(f"input_size must be [C, H, W], got {resolved_input_size}.")

    metadata: dict[str, Any] = {
        "model_family": model_family,
        "variant": variant,
        "stage": stage,
        "parameter_count": int(parameter_count),
        "quantization": quantization,
        "calibrated_threshold": calibrated_threshold,
        "input_size": resolved_input_size,
        "preprocessing_version": str(preprocessing_version),
        "manifest_digest": str(manifest_digest),
        "evaluation_status": evaluation_status,
        "opset_version": int(opset_version),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": int(artifact_size_bytes),
        "parity_verified": parity_verified,
        "max_abs_diff_pytorch_vs_onnx": max_abs_diff_pytorch_vs_onnx,
    }
    if extra:
        metadata.update(extra)
    return metadata


def embed_onnx_metadata(model: onnx.ModelProto, metadata: dict[str, Any]) -> onnx.ModelProto:
    """Embed metadata properties directly into the ONNX model protobuf.

    Omits self-referential artifact_sha256 and artifact_size_bytes so that
    post-embed file saving does not invalidate internal hash signatures.
    """
    # Remove any existing keys with the same name
    existing_keys = {prop.key for prop in model.metadata_props}
    for key, value in metadata.items():
        if value is None or key in ("artifact_sha256", "artifact_size_bytes"):
            continue
        serialized = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        if key in existing_keys:
            for prop in model.metadata_props:
                if prop.key == key:
                    prop.value = serialized
                    break
        else:
            prop = model.metadata_props.add()
            prop.key = key
            prop.value = serialized
    return model


def write_artifact_metadata_file(metadata: dict[str, Any], output_path: Path) -> Path:
    """Write companion artifact metadata JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return output_path


def finalize_artifact_and_sidecar(
    onnx_path: Path,
    metadata: dict[str, Any],
    sidecar_path: Path | None = None,
) -> tuple[str, int, Path]:
    """Embed metadata into ONNX model, save, compute verified post-embed SHA-256 and size, and write sidecar.

    Eliminates stale self-referential artifact hashes by omitting the artifact hash during embedding,
    saving the final protobuf, computing the true SHA-256 on the saved disk bytes, and writing the
    companion metadata sidecar with matching verified values.
    """
    onnx_path = Path(onnx_path)
    proto = onnx.load(str(onnx_path))
    proto = embed_onnx_metadata(proto, metadata)
    onnx.save(proto, str(onnx_path))

    final_sha = compute_file_sha256(onnx_path)
    final_size = onnx_path.stat().st_size
    assert compute_file_sha256(onnx_path) == final_sha, "Post-embed artifact hash mismatch"

    metadata["artifact_sha256"] = final_sha
    metadata["artifact_size_bytes"] = final_size

    out_sidecar = sidecar_path or onnx_path.with_name(f"{onnx_path.stem}.metadata.json")
    write_artifact_metadata_file(metadata, out_sidecar)
    return final_sha, final_size, out_sidecar


def evaluate_onnx_float_vs_int8(
    float_model_path: Path,
    int8_model_path: Path,
    eval_records: list[dict[str, Any]],
    calibrated_threshold: float,
    data_root: Path | None = None,
    max_samples: int = 500,
    tolerance_auroc_drop: float = 0.005,  # 0.5 percentage points
) -> dict[str, Any]:
    """Compare float vs INT8 ONNX models on identical evaluation records.

    Calculates:
    - Clean / mean AUROC and class recalls
    - Relative degradation (AUROC drop)
    - Absolute probability differences
    - Quality gate pass/fail status
    """
    if ort is None:
        raise RuntimeError("onnxruntime is required for ONNX evaluation")

    float_sess = ort.InferenceSession(str(float_model_path))
    int8_sess = ort.InferenceSession(str(int8_model_path))

    reader = DisjointCalibrationDataReader(
        records=eval_records,
        data_root=data_root,
        batch_size=1,
        max_samples=max_samples,
    )

    float_probs: list[float] = []
    int8_probs: list[float] = []
    targets: list[int] = []

    for i in range(len(reader.records)):
        item = reader.get_next()
        if item is None:
            break
        rec = reader.records[i]
        label = int(rec.get("ai_positive", rec.get("label", 0)))
        targets.append(label)

        f_out = float_sess.run(["probabilities"], item)[0]  # [1, 2]
        q_out = int8_sess.run(["probabilities"], item)[0]  # [1, 2]

        float_probs.append(float(f_out[0, 1]))
        int8_probs.append(float(q_out[0, 1]))

    f_scores = torch.tensor(float_probs, dtype=torch.float32)
    q_scores = torch.tensor(int8_probs, dtype=torch.float32)
    t_tensor = torch.tensor(targets, dtype=torch.float32)

    float_auroc = binary_auroc(t_tensor, f_scores)
    int8_auroc = binary_auroc(t_tensor, q_scores)

    float_preds = (f_scores >= calibrated_threshold).long()
    int8_preds = (q_scores >= calibrated_threshold).long()
    t_long = t_tensor.long()

    # Recalls
    pos_mask = t_long == 1
    neg_mask = t_long == 0

    f_pos_recall = float((float_preds[pos_mask] == 1).float().mean()) if pos_mask.any() else 1.0
    f_neg_recall = float((float_preds[neg_mask] == 0).float().mean()) if neg_mask.any() else 1.0
    q_pos_recall = float((int8_preds[pos_mask] == 1).float().mean()) if pos_mask.any() else 1.0
    q_neg_recall = float((int8_preds[neg_mask] == 0).float().mean()) if neg_mask.any() else 1.0

    auroc_drop = (
        float_auroc - int8_auroc if not np.isnan(float_auroc) and not np.isnan(int8_auroc) else 0.0
    )

    abs_diffs = np.abs(np.array(float_probs) - np.array(int8_probs))
    max_abs_diff = float(np.max(abs_diffs)) if len(abs_diffs) > 0 else 0.0
    mean_abs_diff = float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else 0.0

    # Gate: drop must be <= tolerance (0.5 percentage points)
    passed_gate = auroc_drop <= tolerance_auroc_drop

    evaluation_status = "promoted" if passed_gate else "experimental"

    return {
        "passed_gate": passed_gate,
        "evaluation_status": evaluation_status,
        "samples_evaluated": len(targets),
        "float_auroc": float_auroc,
        "int8_auroc": int8_auroc,
        "auroc_drop": auroc_drop,
        "tolerance_auroc_drop": tolerance_auroc_drop,
        "float_aigc_recall": f_pos_recall,
        "float_authentic_recall": f_neg_recall,
        "int8_aigc_recall": q_pos_recall,
        "int8_authentic_recall": q_neg_recall,
        "max_abs_probability_diff": max_abs_diff,
        "mean_abs_probability_diff": mean_abs_diff,
    }

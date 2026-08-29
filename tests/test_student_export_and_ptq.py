"""Focused test suite for student ONNX opset 17 export, static INT8 PTQ, and graph verification.

Validates:
1. Exact checkpoint identity loading & teacher/cross-track rejection guards.
2. Disjoint calibration dataset enforcement (exactly 4096 rows, strict split isolation).
3. PyTorch vs ONNX opset 17 numerical parity.
4. Machine-verifiable static INT8 graph verification (proving QDQ, rejecting MatMulNBits/dynamic).
5. Artifact metadata contract emission and embedding.
6. Promotion gate evaluation tolerances and fallback behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_onnx_pkg = _ROOT / ".runtime" / "onnx_packages"
if _onnx_pkg.exists() and str(_onnx_pkg) not in sys.path:
    sys.path.insert(0, str(_onnx_pkg))
_src = _ROOT / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
import onnx
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper

from aigc_detector.model import ProvenanceHead
from aigc_detector.static_int8 import (
    DisjointCalibrationDataReader,
    build_artifact_metadata,
    compute_file_sha256,
    embed_onnx_metadata,
    evaluate_onnx_float_vs_int8,
    get_nodes_to_exclude_from_quantization,
    inspect_and_verify_static_int8,
    verify_calibration_disjointness,
    write_artifact_metadata_file,
)
from scripts.calibrate_quantize_int8 import (
    calibrate_and_quantize_int8,
    extract_metadata_from_onnx,
)
from scripts.export_onnx_webgpu import (
    StudentWebGPUExportWrapper,
    resolve_promotion_metadata,
    resolve_student_config,
    validate_student_checkpoint,
    verify_pytorch_vs_onnx_parity,
)


class MockStudentModel(nn.Module):
    """Synthetic student detector model matching ViT token_adapter + ProvenanceHead pipeline."""

    def __init__(self, encoder_dim: int = 384, trunk_dim: int = 512) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, encoder_dim, kernel_size=16, stride=16)
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, trunk_dim),
        )
        self.heads = ProvenanceHead(encoder_dim=trunk_dim, trunk_dim=trunk_dim)

    def forward_tensor(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.conv(pixel_values)
        tokens = x.flatten(2).transpose(1, 2)
        adapted = self.token_adapter(tokens)
        aigc_features, tamper_features, _ = self.heads.forward_batched_tokens(adapted)
        ai_positive_features = torch.cat((aigc_features, tamper_features), dim=-1)
        return self.heads.ai_positive_classifier(ai_positive_features).squeeze(-1)


def make_real_image(path: Path) -> Path:
    """Create a real temporary JPEG image on disk for calibration/eval testing."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# 1. Checkpoint identity & track guard tests
# ---------------------------------------------------------------------------


def test_validate_student_checkpoint_small_success(tmp_path: Path) -> None:
    """Valid small student checkpoint (dim 384) passes validation for variant='small'."""
    ckpt_path = tmp_path / "student_small_distill.pt"
    payload = {
        "step": 1000,
        "token_adapter": {
            "0.weight": torch.randn(384),
            "1.weight": torch.randn(512, 384),
        },
        "config": {"model": {"encoder_dim": 384}},
    }
    torch.save(payload, ckpt_path)

    dim = validate_student_checkpoint(ckpt_path, "small")
    assert dim == 384


def test_validate_student_checkpoint_base_success(tmp_path: Path) -> None:
    """Valid base student checkpoint (dim 768) passes validation for variant='base'."""
    ckpt_path = tmp_path / "student_base_post_att.pt"
    payload = {
        "step": 1000,
        "token_adapter": {
            "0.weight": torch.randn(768),
            "1.weight": torch.randn(512, 768),
        },
        "config": {"model": {"encoder_dim": 768}},
    }
    torch.save(payload, ckpt_path)

    dim = validate_student_checkpoint(ckpt_path, "base")
    assert dim == 768


def test_validate_student_checkpoint_rejects_teacher(tmp_path: Path) -> None:
    """Teacher checkpoints (dim 1280 or teacher name) are strictly rejected with clear error."""
    ckpt_path = tmp_path / "teacher_iteration1_checkpoint2.pt"
    payload = {
        "step": 5000,
        "token_adapter": {
            "0.weight": torch.randn(1280),
            "1.weight": torch.randn(512, 1280),
        },
    }
    torch.save(payload, ckpt_path)

    with pytest.raises(ValueError, match="teacher track"):
        validate_student_checkpoint(ckpt_path, "small")

    # Also test teacher dimension even with benign name
    benign_name_ckpt = tmp_path / "weights.pt"
    torch.save(payload, benign_name_ckpt)
    with pytest.raises(ValueError, match="teacher/large backbone"):
        validate_student_checkpoint(benign_name_ckpt, "small")


def test_validate_student_checkpoint_rejects_cross_track(tmp_path: Path) -> None:
    """Requesting small with base checkpoint or base with small checkpoint strictly fails."""
    base_ckpt = tmp_path / "student_base_model.pt"
    payload = {
        "step": 1000,
        "token_adapter": {
            "0.weight": torch.randn(768),
            "1.weight": torch.randn(512, 768),
        },
    }
    torch.save(payload, base_ckpt)

    with pytest.raises(ValueError, match="Cross-track checkpoint loading is strictly forbidden"):
        validate_student_checkpoint(base_ckpt, "small")

    small_ckpt = tmp_path / "student_small_model.pt"
    payload_small = {
        "step": 1000,
        "token_adapter": {
            "0.weight": torch.randn(384),
            "1.weight": torch.randn(512, 384),
        },
    }
    torch.save(payload_small, small_ckpt)

    with pytest.raises(ValueError, match="Cross-track checkpoint loading is strictly forbidden"):
        validate_student_checkpoint(small_ckpt, "base")


def test_validate_student_checkpoint_invalid_variant(tmp_path: Path) -> None:
    """Invalid variant string raises ValueError."""
    ckpt = tmp_path / "test.pt"
    torch.save({"step": 1}, ckpt)
    with pytest.raises(ValueError, match="Requested variant must be 'small' or 'base'"):
        validate_student_checkpoint(ckpt, "large")


# ---------------------------------------------------------------------------
# 2. Calibration disjointness guard tests
# ---------------------------------------------------------------------------


def test_calibration_disjointness_clean(tmp_path: Path) -> None:
    """Exactly 4096 calibration rows disjoint from canonical splits pass guard."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    # Create canonical train and val splits
    train_records = [
        {"row_id": f"train_{i}", "image_path": f"img_train_{i}.jpg", "sha256": f"sha_train_{i}"}
        for i in range(100)
    ]
    val_records = [
        {"row_id": f"val_{i}", "image_path": f"img_val_{i}.jpg", "sha256": f"sha_val_{i}"}
        for i in range(50)
    ]

    (splits_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_records) + "\n", encoding="utf-8"
    )
    (splits_dir / "validation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in val_records) + "\n", encoding="utf-8"
    )

    # Create 4096 disjoint calibration records
    calib_records = [
        {"row_id": f"calib_{i}", "image_path": f"img_calib_{i}.jpg", "sha256": f"sha_calib_{i}"}
        for i in range(4096)
    ]

    report = verify_calibration_disjointness(
        calibration_records=calib_records,
        canonical_manifests_dir=splits_dir,
        allow_non_4096=False,
    )
    assert report.passed
    assert report.calibration_row_count == 4096
    assert len(report.row_id_collisions) == 0
    assert len(report.rejection_reasons) == 0


def test_calibration_disjointness_rejects_wrong_count(tmp_path: Path) -> None:
    """Non-4096 row counts fail when allow_non_4096 is False."""
    calib_records = [
        {"row_id": f"c_{i}", "image_path": f"i_{i}.jpg", "sha256": f"s_{i}"} for i in range(4000)
    ]
    report = verify_calibration_disjointness(
        calib_records, canonical_manifests_dir=None, allow_non_4096=False
    )
    assert not report.passed
    assert any("exactly 4096 rows" in r for r in report.rejection_reasons)

    # But passes if allow_non_4096=True (test override)
    report_allowed = verify_calibration_disjointness(
        calib_records, canonical_manifests_dir=None, allow_non_4096=True
    )
    assert report_allowed.passed


def test_calibration_disjointness_rejects_collisions(tmp_path: Path) -> None:
    """Collisions in row_id, image_path, or sha256 fail closed with diagnostics."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()

    train_records = [
        {"row_id": "collision_row", "image_path": "collision_path.jpg", "sha256": "collision_sha"}
    ]
    (splits_dir / "train.jsonl").write_text(json.dumps(train_records[0]) + "\n", encoding="utf-8")

    # Calibration has colliding row
    calib_records = [
        {"row_id": "collision_row", "image_path": "other_path.jpg", "sha256": "other_sha"},
        {"row_id": "clean_row", "image_path": "collision_path.jpg", "sha256": "other_sha2"},
        {"row_id": "clean_row2", "image_path": "clean_path.jpg", "sha256": "collision_sha"},
    ]

    report = verify_calibration_disjointness(
        calib_records, canonical_manifests_dir=splits_dir, allow_non_4096=True
    )
    assert not report.passed
    assert "collision_row" in report.row_id_collisions
    assert "collision_path.jpg" in report.image_path_collisions
    assert "collision_sha" in report.sha256_collisions
    assert len(report.rejection_reasons) >= 3


# ---------------------------------------------------------------------------
# 3. ONNX opset 17 export & PyTorch parity tests
# ---------------------------------------------------------------------------


def test_student_export_opset17_and_parity(tmp_path: Path) -> None:
    """Export student model to opset 17 and verify PyTorch vs ONNX float parity."""
    model = MockStudentModel(encoder_dim=384, trunk_dim=512).eval()
    wrapper = StudentWebGPUExportWrapper(model).eval()

    onnx_path = tmp_path / "student_small_float32.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        dynamo=False,
    )

    proto = onnx.load(str(onnx_path))
    assert proto.opset_import[0].version == 17

    # Verify input/output signatures
    assert proto.graph.input[0].name == "pixel_values"
    assert proto.graph.output[0].name == "probabilities"

    # Parity verification
    parity = verify_pytorch_vs_onnx_parity(wrapper, onnx_path, tolerance=1e-4, num_samples=3)
    assert parity["passed"]
    assert parity["max_abs_diff"] < 1e-4


# ---------------------------------------------------------------------------
# 4. Static INT8 PTQ & Graph Verification tests
# ---------------------------------------------------------------------------


def test_static_int8_ptq_and_verification(tmp_path: Path) -> None:
    """Quantize to static INT8 and verify graph structure and size reduction."""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    model = MockStudentModel(encoder_dim=384, trunk_dim=512).eval()
    wrapper = StudentWebGPUExportWrapper(model).eval()

    float_path = tmp_path / "model_float.onnx"
    int8_path = tmp_path / "model_int8.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(float_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        dynamo=False,
    )

    float_proto = onnx.load(str(float_path))
    nodes_to_exclude = get_nodes_to_exclude_from_quantization(float_proto)

    img_dir = tmp_path / "calib_imgs"
    img_dir.mkdir()
    calib_records = []
    for i in range(8):
        p = make_real_image(img_dir / f"dummy_{i}.jpg")
        calib_records.append({"image_path": str(p)})
    reader = DisjointCalibrationDataReader(records=calib_records, data_root=None, batch_size=1)

    quantize_static(
        model_input=str(float_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        nodes_to_exclude=nodes_to_exclude,
    )

    # Verify graph structure
    result = inspect_and_verify_static_int8(int8_path)
    assert result.passed
    assert result.static_int8_verified
    assert result.quantization_type == "static_int8"
    assert not result.int4_detected
    assert not result.dynamic_only_detected
    assert result.qdq_node_count > 0
    assert result.matmul_nbits_count == 0
    assert len(result.rejection_reasons) == 0

    # Verify sensitive ops preserved
    assert "Softmax" in result.preserved_sensitive_ops
    assert "Sigmoid" in result.preserved_sensitive_ops


def test_calibration_data_reader_raises_on_missing_images(tmp_path: Path) -> None:
    """DisjointCalibrationDataReader strictly raises FileNotFoundError when images are missing."""
    records = [{"image_path": str(tmp_path / "non_existent_image.jpg")}]
    reader = DisjointCalibrationDataReader(records=records, data_root=None, batch_size=1)
    with pytest.raises(FileNotFoundError, match="Calibration image file not found on disk"):
        reader.get_next()


def test_calibration_data_reader_raises_on_missing_image_path_key() -> None:
    """DisjointCalibrationDataReader strictly raises ValueError when record has no image_path."""
    records = [{"foo": "bar"}]
    reader = DisjointCalibrationDataReader(records=records, data_root=None, batch_size=1)
    with pytest.raises(ValueError, match="missing 'image_path'"):
        reader.get_next()


# ---------------------------------------------------------------------------
# 5. Graph verification anti-INT4 & anti-dynamic guard tests
# ---------------------------------------------------------------------------


def test_inspect_rejects_int4_matmul_nbits(tmp_path: Path) -> None:
    """Graph containing MatMulNBits or 4-bit tensors is strictly rejected."""
    # Build a minimal ONNX graph containing a MatMulNBits node
    node = helper.make_node(
        "MatMulNBits",
        inputs=["A", "B", "scales"],
        outputs=["Y"],
        K=64,
        N=64,
        bits=4,
        block_size=32,
    )
    graph = helper.make_graph(
        [node],
        "test_int4_graph",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    int4_path = tmp_path / "model_int4.onnx"
    onnx.save(model, str(int4_path))

    result = inspect_and_verify_static_int8(int4_path)
    assert not result.passed
    assert result.int4_detected
    assert result.quantization_type == "int4_nbits"
    assert result.matmul_nbits_count == 1
    assert any("MatMulNBits" in r for r in result.rejection_reasons)


def test_inspect_rejects_dynamic_quant_only(tmp_path: Path) -> None:
    """Graph containing DynamicQuantizeLinear without static QDQ is strictly rejected."""
    node = helper.make_node(
        "DynamicQuantizeLinear",
        inputs=["x"],
        outputs=["y", "y_scale", "y_zero_point"],
    )
    graph = helper.make_graph(
        [node],
        "test_dynamic_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 64])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    dyn_path = tmp_path / "model_dynamic.onnx"
    onnx.save(model, str(dyn_path))

    result = inspect_and_verify_static_int8(dyn_path)
    assert not result.passed
    assert result.dynamic_only_detected
    assert result.quantization_type == "dynamic_int8"
    assert any("DynamicQuantizeLinear" in r for r in result.rejection_reasons)


def test_inspect_rejects_unquantized_float(tmp_path: Path) -> None:
    """Unquantized float graph without any quantization ops is rejected."""
    node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph(
        [node],
        "test_float_graph",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 64]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 64]),
        ],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 64])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    float_path = tmp_path / "model_unquantized.onnx"
    onnx.save(model, str(float_path))

    result = inspect_and_verify_static_int8(float_path)
    assert not result.passed
    assert result.quantization_type == "float32"
    assert any("no static INT8 nodes" in r for r in result.rejection_reasons)


# ---------------------------------------------------------------------------
# 6. Artifact Metadata Contract tests
# ---------------------------------------------------------------------------


def test_artifact_metadata_contract(tmp_path: Path) -> None:
    """Artifact metadata strictly satisfies the contract and embeds into ONNX."""
    metadata = build_artifact_metadata(
        model_family="dinov3_vits16",
        variant="small",
        stage="post_att",
        parameter_count=25100000,
        quantization="static_int8",
        calibrated_threshold=0.485,
        input_size=[3, 224, 224],
        preprocessing_version="2",
        manifest_digest="test_digest_123",
        evaluation_status="promoted",
        opset_version=17,
        checkpoint_path="outputs/checkpoint-promoted.pt",
        checkpoint_sha256="abc123sha",
        artifact_path="outputs/student_small_post_att_static_int8.onnx",
        artifact_sha256="def456sha",
        artifact_size_bytes=26000000,
        parity_verified=True,
    )

    # Check contract fields
    contract_fields = [
        "model_family",
        "parameter_count",
        "quantization",
        "calibrated_threshold",
        "input_size",
        "preprocessing_version",
        "manifest_digest",
        "evaluation_status",
    ]
    for f in contract_fields:
        assert f in metadata, f"Missing required contract field: {f}"

    # Test file writing
    meta_file = tmp_path / "test.metadata.json"
    write_artifact_metadata_file(metadata, meta_file)
    assert meta_file.exists()
    loaded = json.loads(meta_file.read_text(encoding="utf-8"))
    assert loaded["model_family"] == "dinov3_vits16"
    assert loaded["parameter_count"] == 25100000

    # Test embedding into ONNX
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "test_meta_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 10])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_model = embed_onnx_metadata(onnx_model, metadata)

    recovered = extract_metadata_from_onnx(onnx_model)
    assert recovered["model_family"] == "dinov3_vits16"
    assert recovered["quantization"] == "static_int8"
    assert recovered["evaluation_status"] == "promoted"

    # Rejects missing manifest_digest
    with pytest.raises(ValueError, match="manifest_digest is required"):
        build_artifact_metadata(
            model_family="dinov3_vits16",
            variant="small",
            stage="post_att",
            parameter_count=25100000,
            quantization="static_int8",
            calibrated_threshold=0.485,
            manifest_digest="",
            evaluation_status="promoted",
        )

    # Rejects invalid quantization
    with pytest.raises(ValueError, match="quantization must be 'float32' or 'static_int8'"):
        build_artifact_metadata(
            model_family="dinov3_vits16",
            variant="small",
            stage="post_att",
            parameter_count=25100000,
            quantization="int4",
            calibrated_threshold=0.485,
            manifest_digest="test_digest",
            evaluation_status="promoted",
        )

    # Rejects invalid evaluation_status
    with pytest.raises(ValueError, match="evaluation_status must be"):
        build_artifact_metadata(
            model_family="dinov3_vits16",
            variant="small",
            stage="post_att",
            parameter_count=25100000,
            quantization="static_int8",
            calibrated_threshold=0.485,
            manifest_digest="test_digest",
            evaluation_status="fake_status",
        )

    # Rejects threshold out of bounds
    with pytest.raises(ValueError, match="calibrated_threshold must be between"):
        build_artifact_metadata(
            model_family="dinov3_vits16",
            variant="small",
            stage="post_att",
            parameter_count=25100000,
            quantization="static_int8",
            calibrated_threshold=1.5,
            manifest_digest="test_digest",
            evaluation_status="promoted",
        )


# ---------------------------------------------------------------------------
# 7. Evaluation & Promotion Gate tests
# ---------------------------------------------------------------------------


def test_evaluate_onnx_float_vs_int8_promotion_gate(tmp_path: Path) -> None:
    """When degradation is within 0.5% AUROC, status is promoted; otherwise experimental."""
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    model = MockStudentModel(encoder_dim=384, trunk_dim=512).eval()
    wrapper = StudentWebGPUExportWrapper(model).eval()

    float_path = tmp_path / "m_float.onnx"
    int8_path = tmp_path / "m_int8.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(float_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        dynamo=False,
    )

    float_proto = onnx.load(str(float_path))
    nodes_to_exclude = get_nodes_to_exclude_from_quantization(float_proto)
    calib_records = []
    for i in range(8):
        p = make_real_image(tmp_path / "gate_calib" / f"d_{i}.jpg")
        calib_records.append({"image_path": str(p)})
    reader = DisjointCalibrationDataReader(records=calib_records, data_root=None, batch_size=1)

    quantize_static(
        model_input=str(float_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
        nodes_to_exclude=nodes_to_exclude,
    )

    eval_records = []
    for i in range(20):
        p = make_real_image(tmp_path / "gate_eval" / f"eval_{i}.jpg")
        eval_records.append({"image_path": str(p), "ai_positive": i % 2})

    # Test with generous tolerance (should pass)
    res_pass = evaluate_onnx_float_vs_int8(
        float_model_path=float_path,
        int8_model_path=int8_path,
        eval_records=eval_records,
        calibrated_threshold=0.485,
        tolerance_auroc_drop=1.0,
    )
    assert res_pass["passed_gate"]
    assert res_pass["evaluation_status"] == "promoted"

    # Test with impossible negative tolerance (must fail and label experimental)
    res_fail = evaluate_onnx_float_vs_int8(
        float_model_path=float_path,
        int8_model_path=int8_path,
        eval_records=eval_records,
        calibrated_threshold=0.485,
        tolerance_auroc_drop=-1.0,
    )
    assert not res_fail["passed_gate"]
    assert res_fail["evaluation_status"] == "experimental"


# ---------------------------------------------------------------------------
# 8. Integration Review Fixes & Authoritative Promotion Contract
# ---------------------------------------------------------------------------


def test_resolve_promotion_metadata_passed(tmp_path: Path) -> None:
    """Passed promotion report derives calibrated_threshold, manifest_digest, and promoted status."""
    ckpt_dir = tmp_path / "student_small"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "checkpoint-promoted.pt"
    ckpt.write_text("fake_weights", encoding="utf-8")

    rep = {
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": "fake_sha",
        "manifest_digest": "manifest_sha256_abcdef123456",
        "calibrated_threshold": 0.4825,
        "metrics": {"auroc": 0.98},
        "passed": True,
        "failed_reasons": [],
    }
    rep_file = ckpt_dir / "promotion_report.json"
    rep_file.write_text(json.dumps(rep), encoding="utf-8")

    thresh, digest, status = resolve_promotion_metadata(checkpoint_path=ckpt)
    assert thresh == 0.4825
    assert digest == "manifest_sha256_abcdef123456"
    assert status == "promoted"


def test_resolve_promotion_metadata_failed(tmp_path: Path) -> None:
    """Failed promotion report marks checkpoint as experimental, never promoted."""
    ckpt_dir = tmp_path / "student_small_failed"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "checkpoint-promoted.pt"
    ckpt.write_text("fake_weights", encoding="utf-8")

    rep = {
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": "fake_sha",
        "manifest_digest": "manifest_sha256_abcdef123456",
        "calibrated_threshold": 0.512,
        "metrics": {"auroc": 0.70},
        "passed": False,
        "failed_reasons": ["auroc_too_low"],
    }
    rep_file = ckpt_dir / "promotion_report.json"
    rep_file.write_text(json.dumps(rep), encoding="utf-8")

    thresh, digest, status = resolve_promotion_metadata(checkpoint_path=ckpt)
    assert thresh == 0.512
    assert digest == "manifest_sha256_abcdef123456"
    assert status == "experimental"


def test_resolve_promotion_metadata_rejects_missing_or_empty_digest(tmp_path: Path) -> None:
    """Missing promotion report and empty manifest digest or threshold strictly fails."""
    ckpt = tmp_path / "standalone_ckpt.pt"
    ckpt.write_text("fake", encoding="utf-8")

    # Empty digest
    with pytest.raises(ValueError, match="manifest-digest is empty"):
        resolve_promotion_metadata(
            checkpoint_path=ckpt, explicit_threshold=0.5, explicit_manifest_digest=""
        )

    # Missing threshold
    with pytest.raises(ValueError, match="calibrated-threshold is not provided"):
        resolve_promotion_metadata(
            checkpoint_path=ckpt, explicit_threshold=None, explicit_manifest_digest="sha123"
        )

    # Unearned 'promoted' claim without passed report strictly fails
    with pytest.raises(ValueError, match="Cannot claim evaluation_status='promoted'"):
        resolve_promotion_metadata(
            checkpoint_path=ckpt,
            explicit_threshold=0.5,
            explicit_manifest_digest="sha123",
            explicit_status="promoted",
        )

    # Without report and without explicit promoted status, status is experimental
    t, d, s = resolve_promotion_metadata(
        checkpoint_path=ckpt,
        explicit_threshold=0.5,
        explicit_manifest_digest="sha123",
    )
    assert s == "experimental"
    assert t == 0.5
    assert d == "sha123"


def test_resolve_student_config_canonical_and_rejects_fallback(tmp_path: Path) -> None:
    """Canonical configs are discovered and missing config raises FileNotFoundError."""
    ckpt = tmp_path / "student.pt"
    torch.save({"step": 1}, ckpt)
    # Stage post_att finds configs/att_student_small.yaml
    cfg = resolve_student_config(variant="small", stage="post_att", checkpoint_path=ckpt)
    assert cfg["model"]["encoder_dim"] == 384
    assert cfg["preprocessing"]["version"] == 2

    # Stage distill finds configs/student_dinov3_small_distill.yaml
    cfg_distill = resolve_student_config(variant="small", stage="distill", checkpoint_path=ckpt)
    assert cfg_distill["model"]["encoder_dim"] == 384

    # Non-existent variant fails with FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Canonical student configuration"):
        resolve_student_config(variant="nonexistent", stage="post_att", checkpoint_path=ckpt)


def test_artifact_metadata_contract_preprocessing_v2_and_verified_sha(tmp_path: Path) -> None:
    """Metadata uses preprocessing version 2, omits self-hash from proto, and verifies sidecar hash."""
    metadata = build_artifact_metadata(
        model_family="dinov3_vits16",
        variant="small",
        stage="post_att",
        parameter_count=25100000,
        quantization="static_int8",
        calibrated_threshold=0.48,
        manifest_digest="manifest_sha_999",
        evaluation_status="promoted",
    )
    assert metadata["preprocessing_version"] == "2"

    # Embed into proto
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "test_sha_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 10])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_model = embed_onnx_metadata(onnx_model, metadata)

    # Self-hash must be omitted from embedded proto to prevent stale hash
    proto_keys = [p.key for p in onnx_model.metadata_props]
    assert "artifact_sha256" not in proto_keys
    assert "artifact_size_bytes" not in proto_keys

    # Save model and verify post-embed hash computation
    saved_onnx = tmp_path / "model_saved.onnx"
    onnx.save(onnx_model, str(saved_onnx))
    final_sha = compute_file_sha256(saved_onnx)
    assert len(final_sha) == 64

    metadata["artifact_sha256"] = final_sha
    metadata["artifact_size_bytes"] = saved_onnx.stat().st_size
    sidecar = tmp_path / "model_saved.metadata.json"
    write_artifact_metadata_file(metadata, sidecar)

    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_data["artifact_sha256"] == final_sha
    assert sidecar_data["preprocessing_version"] == "2"


def test_calibrate_quantize_requires_float_metadata(tmp_path: Path) -> None:
    """PTQ strictly requires float metadata sidecar and raises FileNotFoundError if missing."""
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "float_no_meta",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 10])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 10])],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    float_path = tmp_path / "model_without_meta.onnx"
    onnx.save(onnx_model, str(float_path))

    calib_file = tmp_path / "calib.jsonl"
    calib_file.write_text(json.dumps({"image_path": "img.jpg"}) + "\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Authoritative float metadata sidecar not found"):
        calibrate_and_quantize_int8(
            input_model_path=float_path,
            calibration_manifest_path=calib_file,
            allow_non_4096=True,
        )


def test_calibrate_quantize_missing_eval_manifest_marks_experimental(tmp_path: Path) -> None:
    """Missing eval manifest sets evaluation_status to experimental, never promoted."""

    model = MockStudentModel(encoder_dim=384, trunk_dim=512).eval()
    wrapper = StudentWebGPUExportWrapper(model).eval()

    float_path = tmp_path / "student_small_float.onnx"
    int8_path = tmp_path / "student_small_int8.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(float_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        dynamo=False,
    )

    # Create companion float metadata
    float_meta = build_artifact_metadata(
        model_family="dinov3_vits16",
        variant="small",
        stage="post_att",
        parameter_count=25100000,
        quantization="float32",
        calibrated_threshold=0.485,
        manifest_digest="verified_manifest_sha",
        evaluation_status="promoted",
        preprocessing_version="2",
    )
    write_artifact_metadata_file(
        float_meta, float_path.with_name(f"{float_path.stem}.metadata.json")
    )

    calib_records = []
    for i in range(8):
        p = make_real_image(tmp_path / "ptq_calib" / f"c_{i}.jpg")
        calib_records.append({"image_path": str(p)})
    calib_file = tmp_path / "calibration.jsonl"
    calib_file.write_text("\n".join(json.dumps(r) for r in calib_records) + "\n", encoding="utf-8")

    # Run PTQ WITHOUT eval manifest
    out_model, out_meta, meta_dict = calibrate_and_quantize_int8(
        input_model_path=float_path,
        calibration_manifest_path=calib_file,
        output_model_path=int8_path,
        allow_non_4096=True,
        eval_manifest_path=None,  # Missing eval manifest
    )

    # Must be experimental, never promoted
    assert meta_dict["evaluation_status"] == "experimental"
    assert not meta_dict["gate_passed"]
    assert meta_dict["preprocessing_version"] == "2"
    assert meta_dict["calibrated_threshold"] == 0.485
    assert meta_dict["manifest_digest"] == "verified_manifest_sha"

    # Verify post-embed hash
    final_sha = compute_file_sha256(out_model)
    assert meta_dict["artifact_sha256"] == final_sha


def test_export_skipping_parity_rejects_promotion(tmp_path: Path) -> None:
    """Exporting with skip_parity_check=True cannot earn promoted status."""
    from scripts.export_onnx_webgpu import export_student_to_onnx

    ckpt = tmp_path / "student_small.pt"
    torch.save(
        {
            "step": 1,
            "token_adapter": {"0.weight": torch.randn(384), "1.weight": torch.randn(512, 384)},
            "config": {"model": {"encoder_dim": 384}},
        },
        ckpt,
    )
    rep = {
        "checkpoint_path": str(ckpt),
        "manifest_digest": "digest123",
        "calibrated_threshold": 0.49,
        "passed": True,
    }
    rep_file = tmp_path / "promotion_report.json"
    rep_file.write_text(json.dumps(rep), encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot promote model when --skip-parity-check is set"):
        export_student_to_onnx(
            checkpoint_path=ckpt,
            variant="small",
            output_path=tmp_path / "out.onnx",
            promotion_report_path=rep_file,
            skip_parity_check=True,
        )


def test_calibrate_quantize_unpromoted_float_stays_experimental(tmp_path: Path) -> None:
    """If input float model is experimental, static INT8 model remains experimental even if eval passes."""
    model = MockStudentModel(encoder_dim=384, trunk_dim=512).eval()
    wrapper = StudentWebGPUExportWrapper(model).eval()

    float_path = tmp_path / "float_exp.onnx"
    int8_path = tmp_path / "int8_exp.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(float_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        dynamo=False,
    )

    float_meta = build_artifact_metadata(
        model_family="dinov3_vits16",
        variant="small",
        stage="post_att",
        parameter_count=25100000,
        quantization="float32",
        calibrated_threshold=0.485,
        manifest_digest="verified_manifest_sha",
        evaluation_status="experimental",  # Float is experimental!
        preprocessing_version="2",
    )
    write_artifact_metadata_file(
        float_meta, float_path.with_name(f"{float_path.stem}.metadata.json")
    )

    calib_records = []
    for i in range(8):
        p = make_real_image(tmp_path / "calib_unprom" / f"c_{i}.jpg")
        calib_records.append({"image_path": str(p)})
    calib_file = tmp_path / "calib_unprom.jsonl"
    calib_file.write_text("\n".join(json.dumps(r) for r in calib_records) + "\n", encoding="utf-8")

    eval_records = []
    for i in range(8):
        p = make_real_image(tmp_path / "eval_unprom" / f"e_{i}.jpg")
        eval_records.append({"image_path": str(p), "ai_positive": i % 2})
    eval_file = tmp_path / "eval_unprom.jsonl"
    eval_file.write_text("\n".join(json.dumps(r) for r in eval_records) + "\n", encoding="utf-8")

    out_model, out_meta, meta_dict = calibrate_and_quantize_int8(
        input_model_path=float_path,
        calibration_manifest_path=calib_file,
        output_model_path=int8_path,
        allow_non_4096=True,
        eval_manifest_path=eval_file,
        tolerance_auroc_drop=10.0,  # Gate definitely passes
    )

    assert meta_dict["evaluation_status"] == "experimental"

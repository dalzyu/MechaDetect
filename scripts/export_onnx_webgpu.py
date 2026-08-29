#!/usr/bin/env python3
"""Export selected student checkpoints (ViT-S / ViT-B) to ONNX opset 17 for WebGPU inference.

Generalizes opset 17 export for exact selected student checkpoints, strictly guards
against loading teacher checkpoints or cross-track models, enforces passed promotion reports
or explicit metadata, requires canonical student configs, verifies PyTorch vs ONNX float
parity (failing closed on divergence), and emits the authoritative artifact metadata contract
with verified post-embed SHA-256 sidecars.
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

import numpy as np
import onnxruntime as ort  # Production dependency for export parity verification
import torch
import torch.nn as nn

from aigc_detector.config import load_config
from aigc_detector.predict import _load_checkpoint
from aigc_detector.runtime import load_local_environment
from aigc_detector.static_int8 import (
    build_artifact_metadata,
    compute_file_sha256,
    finalize_artifact_and_sidecar,
)
from aigc_detector.train import build_model
from scripts.distill_student import STUDENT_PRESETS, student_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Model family identifiers
MODEL_FAMILY_MAP = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
}


class StudentWebGPUExportWrapper(nn.Module):
    """Clean inference wrapper for exporting student models to WebGPU/WASM ONNX.

    Accepts preprocessed pixel values tensor [B, 3, 224, 224] (RGB normalized)
    and returns [B, 2] tensor of probabilities where:
      [:, 0] = authentic probability (1 - p_aigc)
      [:, 1] = AI-positive / AIGC probability
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run batched forward pass returning binary [authentic, aigc] probabilities."""
        logit = self.model.forward_tensor(pixel_values)
        p_aigc = torch.sigmoid(logit)
        p_auth = 1.0 - p_aigc
        return torch.stack((p_auth, p_aigc), dim=-1)


def inspect_checkpoint_identity(checkpoint_path: Path) -> dict[str, Any]:
    """Inspect checkpoint file to determine internal weight dimensions and configuration."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    encoder_dim: int | None = None
    config: dict[str, Any] | None = None

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = load_file(str(checkpoint_path), device="cpu")
        for k, v in tensors.items():
            if "token_adapter.0.weight" in k:
                encoder_dim = int(v.shape[0])
                break
            if "token_adapter.1.weight" in k:
                encoder_dim = int(v.shape[1])
                break
            if "patch_embeddings.projection.weight" in k:
                encoder_dim = int(v.shape[0])
                break
        return {"format": "safetensors", "encoder_dim": encoder_dim, "config": None}
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        if "token_adapter" in payload and isinstance(payload["token_adapter"], dict):
            ta = payload["token_adapter"]
            if "0.weight" in ta:
                encoder_dim = int(ta["0.weight"].shape[0])
            elif "1.weight" in ta:
                encoder_dim = int(ta["1.weight"].shape[1])

        if encoder_dim is None and "encoder" in payload and isinstance(payload["encoder"], dict):
            for k, v in payload["encoder"].items():
                if "patch_embeddings.projection.weight" in k:
                    encoder_dim = int(v.shape[0])
                    break

        if encoder_dim is None and config and "model" in config:
            encoder_dim = config["model"].get("encoder_dim")

    return {"format": "pt", "encoder_dim": encoder_dim, "config": config}


def validate_student_checkpoint(
    checkpoint_path: Path,
    requested_variant: str,
) -> int:
    """Validate that the checkpoint is a student model of the requested variant, never teacher or cross-track.

    Strictly enforces:
    - variant must be 'small' or 'base'
    - rejects teacher checkpoints (encoder_dim=1280 or teacher name/path)
    - rejects cross-track loading (requesting small with base weights or vice-versa)
    """
    if requested_variant not in ("small", "base"):
        raise ValueError(f"Requested variant must be 'small' or 'base', got '{requested_variant}'")

    expected_dim = STUDENT_PRESETS[requested_variant]["encoder_dim"]
    other_variant = "base" if requested_variant == "small" else "small"
    other_dim = STUDENT_PRESETS[other_variant]["encoder_dim"]

    # Check filename and parent directory for forbidden teacher clues
    name_lower = checkpoint_path.name.lower()
    parent_lower = str(checkpoint_path.parent).lower().replace("\\", "/")
    if (
        "teacher" in name_lower
        or "checkpoint2" in name_lower
        or "vith16" in name_lower
        or "/teachers/" in parent_lower
    ):
        raise ValueError(
            f"Export rejected: Checkpoint '{checkpoint_path}' is from the teacher track. "
            "scripts/export_onnx_webgpu.py strictly exports student checkpoints (small or base), never teacher models."
        )

    info = inspect_checkpoint_identity(checkpoint_path)
    detected_dim = info.get("encoder_dim")

    if detected_dim in (1280, 1536, 1024):
        raise ValueError(
            f"Export rejected: Checkpoint '{checkpoint_path}' has encoder_dim={detected_dim} "
            "(teacher/large backbone). scripts/export_onnx_webgpu.py strictly exports student checkpoints."
        )

    if detected_dim == other_dim:
        raise ValueError(
            f"Export rejected: Requested variant '{requested_variant}' (expected dim={expected_dim}), "
            f"but checkpoint '{checkpoint_path}' has dim={detected_dim} ('{other_variant}'). "
            "Cross-track checkpoint loading is strictly forbidden."
        )

    if detected_dim is not None and detected_dim != expected_dim:
        raise ValueError(
            f"Export rejected: Unexpected encoder dimension {detected_dim} in checkpoint '{checkpoint_path}' "
            f"(expected {expected_dim} for '{requested_variant}')."
        )

    logger.info(
        "Checkpoint identity verified: variant='%s' (encoder_dim=%d)",
        requested_variant,
        expected_dim,
    )
    return expected_dim


def verify_pytorch_vs_onnx_parity(
    pytorch_wrapper: nn.Module,
    onnx_path: Path,
    tolerance: float = 1e-4,
    num_samples: int = 3,
) -> dict[str, Any]:
    """Verify inference parity between PyTorch float wrapper and exported ONNX model.

    Fails hard if onnxruntime cannot run the model or if numerical divergence exceeds tolerance.
    """
    session = ort.InferenceSession(str(onnx_path))
    pytorch_wrapper.eval()

    # Generate synthetic input batch [num_samples, 3, 224, 224]
    rng = torch.Generator().manual_seed(42)
    sample_inputs = torch.randn(num_samples, 3, 224, 224, generator=rng, dtype=torch.float32)

    with torch.no_grad():
        pt_outputs = pytorch_wrapper(sample_inputs).detach().cpu().numpy()

    onnx_outputs = session.run(["probabilities"], {"pixel_values": sample_inputs.numpy()})[0]

    abs_diff = np.abs(pt_outputs - onnx_outputs)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    passed = max_diff <= tolerance
    if not passed:
        error_msg = (
            f"PyTorch vs ONNX parity check FAILED: max_diff={max_diff:.6e} > tolerance={tolerance:.6e}. "
            "Numerical output diverged beyond acceptable threshold."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(
        "PyTorch vs ONNX parity check PASSED: max_diff=%.6e, mean_diff=%.6e",
        max_diff,
        mean_diff,
    )

    return {
        "passed": True,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": tolerance,
        "num_samples": num_samples,
    }


def resolve_promotion_metadata(
    checkpoint_path: Path,
    promotion_report_path: Path | None = None,
    explicit_threshold: float | None = None,
    explicit_manifest_digest: str | None = None,
    explicit_status: str | None = None,
) -> tuple[float, str, str]:
    """Resolve and validate calibrated_threshold, manifest_digest, and evaluation_status.

    Rejects default 0.5, empty digest, and unearned 'promoted' status unless proven
    by a passed promotion report. Unearned promotion claims without evidence are strictly rejected.
    """
    report_file = promotion_report_path
    if report_file is None:
        # Search candidate locations next to checkpoint
        candidates = [
            checkpoint_path.parent / "promotion_report.json",
            checkpoint_path.parent.parent / "att_shared_promotion_report.json",
        ]
        for cand in candidates:
            if cand.is_file():
                report_file = cand
                break

    if report_file is not None and Path(report_file).is_file():
        logger.info("Resolving metadata from promotion report: %s", report_file)
        with open(report_file, encoding="utf-8") as f:
            report_data = json.load(f)

        passed = bool(report_data.get("passed", False))
        threshold_val = report_data.get("calibrated_threshold")
        digest_val = report_data.get("manifest_digest")

        if threshold_val is None:
            raise ValueError(f"Promotion report '{report_file}' is missing 'calibrated_threshold'.")
        if not digest_val:
            raise ValueError(
                f"Promotion report '{report_file}' contains an empty or missing 'manifest_digest'."
            )

        calibrated_threshold = float(threshold_val)
        manifest_digest = str(digest_val)
        evaluation_status = "promoted" if passed else "experimental"

        if not passed:
            logger.warning(
                "Promotion report '%s' marked checkpoint as FAILED (passed=False). Status set to 'experimental'.",
                report_file,
            )
        else:
            logger.info(
                "Promotion report verified: status='promoted', calibrated_threshold=%.4f, manifest_digest=%s",
                calibrated_threshold,
                manifest_digest[:16],
            )
        return calibrated_threshold, manifest_digest, evaluation_status

    # Fallback: No promotion report found; check explicit CLI inputs
    if not explicit_manifest_digest:
        raise ValueError(
            "Export rejected: No promotion report found and --manifest-digest is empty. "
            "Empty digest and unearned 'promoted' status are strictly forbidden. "
            "Please provide a promotion report or explicit --manifest-digest and --calibrated-threshold."
        )

    if explicit_threshold is None:
        raise ValueError(
            "Export rejected: No promotion report found and --calibrated-threshold is not provided. "
            "Default uncalibrated 0.5 threshold is strictly forbidden."
        )

    calibrated_threshold = float(explicit_threshold)
    manifest_digest = str(explicit_manifest_digest)

    if explicit_status == "promoted":
        raise ValueError(
            "Export rejected: Cannot claim evaluation_status='promoted' without a passed promotion report. "
            "Promotion strictly requires verified evidence from a passed promotion evaluation."
        )

    evaluation_status = explicit_status or "experimental"
    return calibrated_threshold, manifest_digest, evaluation_status


def resolve_student_config(
    variant: str,
    stage: str,
    checkpoint_path: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve canonical student config, eliminating legacy teacher/fallback configs."""
    load_local_environment(REPO_ROOT)
    if config_path is not None and Path(config_path).is_file():
        logger.info("Loading student config from: %s", config_path)
        return load_config(Path(config_path))
    # Priority 1: Canonical ATT student config
    if stage in ("att", "post_att"):
        att_cfg = REPO_ROOT / "configs" / f"att_student_{variant}.yaml"
        if att_cfg.is_file():
            logger.info("Using canonical ATT student config: %s", att_cfg)
            return load_config(att_cfg)

    # Priority 2: Canonical distillation student config
    distill_cfg = REPO_ROOT / "configs" / f"student_dinov3_{variant}_distill.yaml"
    if distill_cfg.is_file():
        logger.info("Using canonical distillation student config: %s", distill_cfg)
        return load_config(distill_cfg)

    # Priority 3: Config embedded in checkpoint payload
    info = inspect_checkpoint_identity(checkpoint_path)
    if info.get("config") is not None:
        logger.info("Using configuration embedded in checkpoint payload.")
        return info["config"]

    raise FileNotFoundError(
        f"Canonical student configuration for variant '{variant}' not found. "
        f"Expected configs/att_student_{variant}.yaml or configs/student_dinov3_{variant}_distill.yaml. "
        "Checkpoint2/legacy fallback configs are forbidden. Please specify --config."
    )


def export_student_to_onnx(
    checkpoint_path: Path,
    variant: str,
    output_path: Path | None = None,
    stage: str = "post_att",
    config_path: Path | None = None,
    promotion_report_path: Path | None = None,
    calibrated_threshold: float | None = None,
    manifest_digest: str | None = None,
    evaluation_status: str | None = None,
    metadata_output: Path | None = None,
    skip_parity_check: bool = False,
    opset_version: int = 17,
) -> tuple[Path, Path, dict[str, Any]]:
    """Export a student checkpoint to ONNX opset 17 and emit artifact metadata."""
    load_local_environment(REPO_ROOT)
    checkpoint_path = Path(checkpoint_path).resolve()

    # 1. Strict architecture and checkpoint validation
    validate_student_checkpoint(checkpoint_path, variant)

    # 2. Resolve promotion metadata (threshold, manifest_digest, status)
    resolved_threshold, resolved_digest, resolved_status = resolve_promotion_metadata(
        checkpoint_path=checkpoint_path,
        promotion_report_path=promotion_report_path,
        explicit_threshold=calibrated_threshold,
        explicit_manifest_digest=manifest_digest,
        explicit_status=evaluation_status,
    )
    if skip_parity_check and resolved_status == "promoted":
        raise ValueError(
            "Export rejected: Cannot promote model when --skip-parity-check is set. "
            "PyTorch vs ONNX parity verification is strictly required for promotion."
        )
    # 3. Resolve canonical student configuration (strictly no checkpoint2 fallback)
    base_config = resolve_student_config(variant, stage, checkpoint_path, config_path)
    config = (
        student_config(base_config, variant)
        if "training" not in base_config or "stage" not in base_config["training"]
        else base_config
    )
    if "model" in config:
        config["model"].setdefault("branch_dim", 256)
        config["model"].setdefault("dropout", 0.1)

    preprocessing_version = str(config.get("preprocessing", {}).get("version", 2))

    # 4. Build model architecture & load weights
    logger.info("Building student model for variant '%s'...", variant)
    model = build_model(config)
    logger.info("Loading checkpoint weights from: %s", checkpoint_path)
    _load_checkpoint(model, checkpoint_path)
    model.eval()

    # Wrap model for WebGPU inference
    wrapper = StudentWebGPUExportWrapper(model).eval()

    # 5. Resolve output path
    if output_path is None:
        output_dir = REPO_ROOT / "outputs" / "exported_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"student_{variant}_{stage}_float32.onnx"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Export to ONNX opset 17
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    logger.info("Exporting ONNX model to: %s (opset=%d)", output_path, opset_version)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        },
        dynamo=False,
    )

    # 7. Verify PyTorch vs ONNX float parity (production dependency; fails hard on divergence)
    if skip_parity_check:
        if resolved_status == "promoted":
            raise ValueError(
                "Export rejected: Cannot promote model when --skip-parity-check is set. "
                "PyTorch vs ONNX parity verification is strictly required for promotion."
            )
        logger.warning("Skipping PyTorch vs ONNX parity check; model marked parity_verified=False.")
        parity_result = {
            "passed": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "skipped": True,
        }
    else:
        parity_result = verify_pytorch_vs_onnx_parity(wrapper, output_path)
    # 8. Compute parameter count and checkpoint hash
    param_count = sum(p.numel() for p in model.parameters())
    checkpoint_sha = compute_file_sha256(checkpoint_path)

    # 9. Build artifact metadata contract
    model_family = MODEL_FAMILY_MAP.get(variant, f"dinov3_{variant}")
    metadata = build_artifact_metadata(
        model_family=model_family,
        variant=variant,
        stage=stage,
        parameter_count=param_count,
        quantization="float32",
        calibrated_threshold=resolved_threshold,
        input_size=[3, 224, 224],
        preprocessing_version=preprocessing_version,
        manifest_digest=resolved_digest,
        evaluation_status=resolved_status,
        opset_version=opset_version,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        artifact_path=str(output_path),
        parity_verified=parity_result.get("passed", False),
        max_abs_diff_pytorch_vs_onnx=parity_result.get("max_abs_diff"),
    )

    # 10. Embed metadata into ONNX model, compute verified post-embed SHA-256 and size, write sidecar
    final_artifact_sha, final_artifact_size, meta_path = finalize_artifact_and_sidecar(
        onnx_path=output_path,
        metadata=metadata,
        sidecar_path=metadata_output,
    )
    logger.info(
        "Export complete: %s (%.2f MB, sha256=%s...)",
        output_path,
        final_artifact_size / 1e6,
        final_artifact_sha[:16],
    )
    logger.info("Artifact metadata written to: %s", meta_path)

    return output_path, meta_path, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export student checkpoints (ViT-S / ViT-B) to WebGPU ONNX opset 17."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to student checkpoint (.pt or .safetensors)",
    )
    parser.add_argument(
        "--variant",
        choices=["small", "base"],
        required=True,
        help="Student architecture variant ('small' for ViT-S, 'base' for ViT-B)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="post_att",
        help="Model lifecycle stage (e.g. distill, att, post_att)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Target ONNX export destination (defaults to collision-proof naming)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--promotion-report",
        type=Path,
        default=None,
        help="Path to promotion_report.json to derive threshold, digest, and status",
    )
    parser.add_argument(
        "--calibrated-threshold",
        type=float,
        default=None,
        help="Explicit decision threshold (required if no promotion report is found)",
    )
    parser.add_argument(
        "--manifest-digest",
        type=str,
        default=None,
        help="Explicit SHA256 manifest digest (required if no promotion report is found)",
    )
    parser.add_argument(
        "--evaluation-status",
        type=str,
        default=None,
        help="Explicit evaluation status (promoted, experimental, candidate)",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Path for companion metadata JSON (defaults to alongside ONNX file)",
    )
    parser.add_argument(
        "--skip-parity-check",
        action="store_true",
        help="Skip PyTorch vs ONNX float numerical parity check",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX operator set version (default: 17)",
    )
    args = parser.parse_args()

    export_student_to_onnx(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        output_path=args.output,
        stage=args.stage,
        config_path=args.config,
        promotion_report_path=args.promotion_report,
        calibrated_threshold=args.calibrated_threshold,
        manifest_digest=args.manifest_digest,
        evaluation_status=args.evaluation_status,
        metadata_output=args.metadata_output,
        skip_parity_check=args.skip_parity_check,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quantize Checkpoint 2 using MatMulNBits (INT4 block-wise quantization) for WebGPU."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn as nn
from aigc_detector.runtime import load_local_environment
from aigc_detector.config import load_config
from aigc_detector.train import build_model
from aigc_detector.predict import _load_checkpoint


class WebGPUModelWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raw_logit = self.model.forward_tensor(pixel_values)
        p_aigc = torch.sigmoid(raw_logit)
        p_auth = 1.0 - p_aigc
        return torch.stack((p_auth, p_aigc), dim=-1)


def main():
    load_local_environment(REPO_ROOT)
    config = load_config(REPO_ROOT / "configs" / "teacher_dinov3_checkpoint2_full_data.yaml")

    print("[1/3] Building model & loading safetensors...")
    model = build_model(config).eval()
    _load_checkpoint(
        model,
        REPO_ROOT / "models" / "teachers" / "iteration1" / "checkpoint2" / "model-weights.safetensors",
    )
    wrapper = WebGPUModelWrapper(model).eval()

    tmp_fp32_dir = REPO_ROOT / "tmp" / "export_fp32"
    tmp_fp32_dir.mkdir(parents=True, exist_ok=True)
    tmp_fp32_onnx = tmp_fp32_dir / "model.onnx"

    print(f"[2/3] Exporting base ONNX model to {tmp_fp32_onnx}...")
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(tmp_fp32_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamo=False,
    )
    print("Base export complete!")

    output_path = REPO_ROOT / "web" / "model" / "checkpoint2.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3/3] Applying MatMulNBits (4-bit block-wise quantization for WebGPU)...")
    # Load onnx model proto
    onnx_model = onnx.load(str(tmp_fp32_onnx), load_external_data=True)
    
    # 4-bit quantization with block size 32 (high accuracy for vision transformers)
    quantizer = MatMulNBitsQuantizer(
        model=onnx_model,
        bits=4,
        block_size=32,
        is_symmetric=True,
    )
    quantizer.process()

    print(f"Saving quantized model to {output_path}...")
    onnx.save_model(quantizer.model.model, str(output_path), save_as_external_data=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Done! Quantized WebGPU model saved: {output_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

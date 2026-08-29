#!/usr/bin/env python3
"""Export Checkpoint 2 to ONNX for pure client-side WebGPU execution."""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aigc_detector.runtime import load_local_environment
from aigc_detector.config import load_config
from aigc_detector.train import build_model
from aigc_detector.predict import _load_checkpoint


class WebGPUModelWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Calls the built-in forward_tensor designed for ONNX/WebGPU
        raw_logit = self.model.forward_tensor(pixel_values)
        p_aigc = torch.sigmoid(raw_logit)
        p_auth = 1.0 - p_aigc
        return torch.stack((p_auth, p_aigc), dim=-1)


def main():
    load_local_environment(REPO_ROOT)
    config = load_config(REPO_ROOT / "configs" / "teacher_dinov3_checkpoint2_full_data.yaml")
    
    print("Building model...")
    model = build_model(config).eval()
    
    print("Loading Checkpoint 2 safetensors...")
    _load_checkpoint(
        model,
        REPO_ROOT / "models" / "teachers" / "iteration1" / "checkpoint2" / "model-weights.safetensors",
    )
    
    wrapper = WebGPUModelWrapper(model).eval()
    
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    output_path = REPO_ROOT / "web" / "model" / "checkpoint2.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to {output_path} (opset 17, dynamo=False)...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["probabilities"],
        dynamo=False,
    )
    print(f"Export complete! Size: {output_path.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()

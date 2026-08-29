#!/usr/bin/env python3
"""Compare native Checkpoint 2 (PyTorch) vs Quantized INT8 (ONNX) on benchmark data."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aigc_detector.runtime import load_local_environment
from aigc_detector.config import load_config
from aigc_detector.train import build_model
from aigc_detector.predict import _load_checkpoint


def main():
    load_local_environment(REPO_ROOT)
    config = load_config(REPO_ROOT / "configs" / "teacher_dinov3_checkpoint2_full_data.yaml")

    print("=" * 75)
    print("  LOADING NATIVE CHECKPOINT 2 (PyTorch FP32/CUDA)...")
    print("=" * 75)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    native_model = build_model(config).to(device).eval()
    _load_checkpoint(
        native_model,
        REPO_ROOT / "models" / "teachers" / "iteration1" / "checkpoint2" / "model-weights.safetensors",
    )
    print(f"Native model ready on {device}!\n")

    print("=" * 75)
    print("  LOADING QUANTIZED INT8 MODEL (ONNX Runtime)...")
    print("=" * 75)
    quant_path = REPO_ROOT / "web" / "model" / "checkpoint2.onnx"
    quant_session = ort.InferenceSession(str(quant_path))
    quant_size_mb = quant_path.stat().st_size / (1024 * 1024)
    print(f"Quantized ONNX session ready! Size: {quant_size_mb:.1f} MB\n")

    # Gather test images
    test_files = []
    
    # Authentic samples
    if (REPO_ROOT / "web" / "samples" / "sample_authentic_imagenet.jpg").exists():
        test_files.append(("Authentic ImageNet", REPO_ROOT / "web" / "samples" / "sample_authentic_imagenet.jpg"))
    if (REPO_ROOT / "tmp" / "artic_audit_00.jpg").exists():
        test_files.append(("Authentic Audit 00", REPO_ROOT / "tmp" / "artic_audit_00.jpg"))
    if (REPO_ROOT / "tmp" / "artic_audit_01.jpg").exists():
        test_files.append(("Authentic Audit 01", REPO_ROOT / "tmp" / "artic_audit_01.jpg"))
    if (REPO_ROOT / "tmp" / "artic_audit_04.jpg").exists():
        test_files.append(("Authentic Audit 04", REPO_ROOT / "tmp" / "artic_audit_04.jpg"))

    # AIGC samples
    if (REPO_ROOT / "tmp" / "artic_audit_02.jpg").exists():
        test_files.append(("AIGC (Audit 02)", REPO_ROOT / "tmp" / "artic_audit_02.jpg"))
    if (REPO_ROOT / "web" / "samples" / "sample_aigc_krea.jpg").exists():
        test_files.append(("AIGC (Krea)", REPO_ROOT / "web" / "samples" / "sample_aigc_krea.jpg"))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    print(f"{'Image Name':<24} | {'Native Logit':<12} | {'Native P(AIGC)':<14} | {'Native Verdict':<14} | {'Quant P(AIGC)':<14} | {'Quant Verdict':<14} | {'Match?'}")
    print("-" * 115)

    differences = []

    for label, img_path in test_files:
        img = Image.open(img_path).convert("RGB")

        # 1. Native PyTorch Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            out_native = native_model([img])
            raw_logit = float(out_native.ai_positive_logit.item())
            native_p_aigc = float(torch.sigmoid(torch.tensor(raw_logit)).item())
        native_ms = (time.perf_counter() - t0) * 1000
        native_is_aigc = native_p_aigc >= 0.5
        native_verdict = f"{'AIGC' if native_is_aigc else 'Original'} {round((native_p_aigc if native_is_aigc else 1-native_p_aigc)*100)}%"

        # 2. Quantized ONNX Inference
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        arr = (arr - mean) / std

        t0 = time.perf_counter()
        out_quant = quant_session.run(None, {"pixel_values": arr})[0]
        quant_ms = (time.perf_counter() - t0) * 1000
        quant_p_auth = float(out_quant[0][0])
        quant_p_aigc = float(out_quant[0][1])
        quant_is_aigc = quant_p_aigc >= 0.5
        quant_verdict = f"{'AIGC' if quant_is_aigc else 'Original'} {round((quant_p_aigc if quant_is_aigc else quant_p_auth)*100)}%"

        diff = abs(native_p_aigc - quant_p_aigc)
        differences.append(diff)
        match = "YES" if (native_is_aigc == quant_is_aigc) else "MISMATCH"

        print(f"{label:<24} | {raw_logit:>+10.3f}   | {native_p_aigc:>12.4f}   | {native_verdict:<14} | {quant_p_aigc:>12.4f}   | {quant_verdict:<14} | {match}")

    print("-" * 115)
    print(f"Average Probability Shift between Native & Quantized: {np.mean(differences):.4f}")


if __name__ == "__main__":
    main()

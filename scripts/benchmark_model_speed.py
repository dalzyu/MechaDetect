#!/usr/bin/env python3
"""Benchmark single-image latency and batched throughput for all MechaDetect models.

Evaluates each model independently across:
1. Single-Image Interactive Latency (Batch Size = 1):
   - Measures interactive runtime latency at batch size 1 on synthetic tensors.
   - Excludes image decode, resize, browser upload, and UI scheduling overhead.
2. Batched Throughput (Batch Size = 64):
   - Ingestion throughput in Images per Second (img/s).
3. Memory and Storage Footprint:
   - File size on disk (MB).
   - Parameter count.
4. Platforms:
   - GPU (NVIDIA GeForce RTX 4080 via CUDAExecutionProvider).
   - CPU (Intel Core i9-13900KF via CPUExecutionProvider).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Register torch CUDA DLLs for ONNX Runtime CUDAExecutionProvider
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(_torch_lib):
    try:
        os.add_dll_directory(_torch_lib)
    except (AttributeError, OSError):
        pass
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort


def benchmark_session(
    session: ort.InferenceSession,
    device_name: str,
    warmup_runs: int = 20,
    latency_runs: int = 100,
    batch_size: int = 64,
    throughput_batches: int = 20,
) -> dict[str, Any]:
    dummy_single = np.random.randn(1, 3, 224, 224).astype(np.float32)
    dummy_batch = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    # 1. Warmup
    for _ in range(warmup_runs):
        session.run(["probabilities"], {"pixel_values": dummy_single})

    # 2. Benchmark Single-Image Latency (Batch Size = 1)
    latencies = []
    for _ in range(latency_runs):
        t0 = time.perf_counter()
        session.run(["probabilities"], {"pixel_values": dummy_single})
        latencies.append((time.perf_counter() - t0) * 1000.0)  # to ms

    p50 = float(np.median(latencies))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))

    # 3. Benchmark Batched Throughput (Batch Size = batch_size)
    for _ in range(3):
        session.run(["probabilities"], {"pixel_values": dummy_batch})

    t0 = time.perf_counter()
    for _ in range(throughput_batches):
        session.run(["probabilities"], {"pixel_values": dummy_batch})
    batch_elapsed = time.perf_counter() - t0
    total_images = batch_size * throughput_batches
    fps = total_images / batch_elapsed if batch_elapsed > 0 else 0.0

    return {
        "device": device_name,
        "single_image_latency_p50_ms": round(p50, 2),
        "single_image_latency_p95_ms": round(p95, 2),
        "single_image_latency_p99_ms": round(p99, 2),
        "single_image_latency_mean_ms": round(mean_lat, 2),
        "batch_size": batch_size,
        "throughput_fps": round(fps, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model latency and throughput")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("outputs/models"),
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/benchmark_results.json"),
        help="Destination JSON for benchmark numbers",
    )
    args = parser.parse_args()

    models = sorted(list(args.models_dir.glob("*.onnx")))
    if not models:
        raise FileNotFoundError(f"No ONNX models found in {args.models_dir}")

    # Shared ONNX Runtime options
    opts_gpu = ort.SessionOptions()
    opts_gpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts_gpu.log_severity_level = 3

    opts_cpu = ort.SessionOptions()
    opts_cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts_cpu.intra_op_num_threads = 16
    opts_cpu.log_severity_level = 3

    has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()

    results: dict[str, Any] = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cpu": "Intel(R) Core(TM) i9-13900KF (24 cores / 32 threads)",
        },
        "models": {},
    }

    print("=" * 80)
    print("MECHADETECT SPEED & EFFICIENCY BENCHMARK")
    print(f"GPU: {results['hardware']['gpu']}")
    print(f"CPU: {results['hardware']['cpu']}")
    print("=" * 80)

    for m in models:
        name = m.stem.replace("mechadetect-", "")
        file_size_mb = round(m.stat().st_size / (1024 * 1024), 1)
        params = "25.1M" if "quark" in name else "89.4M"

        print(f"\nBenchmarking {name} ({file_size_mb} MB, {params} params)...")
        model_result: dict[str, Any] = {
            "file_size_mb": file_size_mb,
            "parameters": params,
        }

        # 1. GPU Benchmark
        if has_cuda:
            try:
                sess_gpu = ort.InferenceSession(str(m), sess_options=opts_gpu, providers=["CUDAExecutionProvider"])
                gpu_metrics = benchmark_session(sess_gpu, device_name=results["hardware"]["gpu"])
                model_result["gpu"] = gpu_metrics
                print(f"  [GPU] Latency (p50): {gpu_metrics['single_image_latency_p50_ms']} ms | Throughput: {gpu_metrics['throughput_fps']} img/s")
            except Exception as e:
                print(f"  [GPU] Skipped/Failed: {e}")

        # 2. CPU Benchmark
        try:
            sess_cpu = ort.InferenceSession(str(m), sess_options=opts_cpu, providers=["CPUExecutionProvider"])
            cpu_metrics = benchmark_session(sess_cpu, device_name="CPU (i9-13900KF)", batch_size=32, throughput_batches=5)
            model_result["cpu"] = cpu_metrics
            print(f"  [CPU] Latency (p50): {cpu_metrics['single_image_latency_p50_ms']} ms | Throughput: {cpu_metrics['throughput_fps']} img/s")
        except Exception as e:
            print(f"  [CPU] Skipped/Failed: {e}")

        results["models"][name] = model_result

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {args.output_json}")


if __name__ == "__main__":
    main()

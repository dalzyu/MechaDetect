#!/usr/bin/env python3
"""Launch one or both Adversarial Transformation Training tracks.

GPU 0 and sequential execution are the defaults. Explicit device pools enable
DDP, and ``--parallel-tracks`` is available only when the small and base pools
are disjoint. Gradient accumulation scales to preserve an effective batch of 48.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


TRACK_SPECS = {
    "small": {"port": 29503, "physical_batch_size": 4},
    "base": {"port": 29504, "physical_batch_size": 2},
}


def _device_ids(value: str) -> tuple[str, ...]:
    devices = tuple(device.strip() for device in value.split(",") if device.strip())
    if not devices:
        raise ValueError("At least one CUDA device must be specified")
    if len(devices) != len(set(devices)):
        raise ValueError(f"Duplicate CUDA devices are not allowed: {value}")
    return devices


def _batch_geometry(variant: str, devices: str, effective_batch_size: int) -> tuple[int, int, int]:
    world_size = len(_device_ids(devices))
    physical_batch_size = int(TRACK_SPECS[variant]["physical_batch_size"])
    divisor = physical_batch_size * world_size
    if effective_batch_size <= 0 or effective_batch_size % divisor:
        raise ValueError(
            f"Effective batch {effective_batch_size} must be divisible by "
            f"{physical_batch_size} physical x {world_size} processes for {variant}"
        )
    return world_size, physical_batch_size, effective_batch_size // divisor


def find_latest_checkpoint(output_dir: Path | str) -> Path | None:
    """Find the latest valid coverage or step checkpoint to resume training."""
    out = Path(output_dir)
    if not out.is_dir():
        return None
    candidates = list(out.glob("checkpoint-coverage-*.pt"))
    if not candidates:
        candidates = [
            p
            for p in out.glob("checkpoint-*.pt")
            if p.is_file()
            and p.name not in ("checkpoint-promoted.pt", "checkpoint-final.pt", "checkpoint-best.pt")
        ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def build_track_command(
    variant: str,
    *,
    checkpoint: Path | str | None,
    manifest: Path | str | None,
    config: Path | str | None,
    output_dir: Path | str,
    num_candidates: int = 3,
    epochs: int = 1,
    port: int = 29503,
    nproc_per_node: int = 1,
    physical_batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    resume: Path | str | None = None,
    dry_run: bool = False,
    python_exe: str = sys.executable,
) -> list[str]:
    """Build one direct or DDP ATT command."""
    script_path = Path(__file__).resolve().parent / "train_att.py"
    if nproc_per_node == 1:
        cmd = [python_exe, str(script_path)]
    else:
        cmd = [
            python_exe,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc_per_node}",
            f"--master_port={port}",
            str(script_path),
        ]
    cmd.extend(
        [
            "--variant",
            variant,
            "--output-dir",
            str(output_dir),
            "--num-candidates",
            str(num_candidates),
            "--epochs",
            str(epochs),
        ]
    )
    cmd.extend(["--world-size", str(nproc_per_node)])
    if checkpoint:
        cmd.extend(["--student-checkpoint", str(checkpoint)])
    if manifest:
        cmd.extend(["--manifest", str(manifest)])
    if config:
        cmd.extend(["--config", str(config)])
    if physical_batch_size is not None:
        cmd.extend(["--batch-size", str(physical_batch_size)])
    if gradient_accumulation is not None:
        cmd.extend(["--gradient-accumulation", str(gradient_accumulation)])
    if resume:
        cmd.extend(["--resume", str(resume)])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def launch_track(
    variant: str,
    *,
    checkpoint: Path | str | None,
    manifest: Path | str | None,
    config: Path | str | None,
    output_dir: Path | str,
    devices: str = "0",
    port: int | None = None,
    num_candidates: int = 3,
    epochs: int = 1,
    effective_batch_size: int = 48,
    resume: Path | str | None = None,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Launch one ATT track on its explicit CUDA device pool."""
    world_size, physical_batch_size, gradient_accumulation = _batch_geometry(
        variant, devices, effective_batch_size
    )
    master_port = port or int(TRACK_SPECS[variant]["port"])
    cmd = build_track_command(
        variant,
        checkpoint=checkpoint,
        manifest=manifest,
        config=config,
        output_dir=output_dir,
        num_candidates=num_candidates,
        epochs=epochs,
        port=master_port,
        nproc_per_node=world_size,
        physical_batch_size=physical_batch_size,
        gradient_accumulation=gradient_accumulation,
        resume=resume,
        dry_run=dry_run,
    )

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["MASTER_PORT"] = str(master_port)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / f"att_launch_{variant}.log"
    log_handle = open(log_file, "w", encoding="utf-8")

    print(
        f"[{variant.upper()}] Launching on GPU pool {devices}: "
        f"{physical_batch_size} physical x {world_size} process(es) x "
        f"{gradient_accumulation} accumulation = {effective_batch_size}"
    )
    print(f"[{variant.upper()}] Log file: {log_file}")
    print(f"[{variant.upper()}] Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    proc.log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch single-GPU ATT by default; multi-GPU is explicit."
    )
    parser.add_argument(
        "--track",
        choices=["small", "base", "both"],
        default="both",
        help="ATT track(s) to run sequentially by default",
    )
    parser.add_argument(
        "--small-checkpoint",
        type=Path,
        default=None,
        help="Path to the final float ViT-S checkpoint",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Path to the final float ViT-B checkpoint",
    )
    parser.add_argument(
        "--train-manifest", type=Path, default=None, help="Path to training manifest"
    )
    parser.add_argument(
        "--small-config",
        type=Path,
        default=Path("configs/att_student_small.yaml"),
        help="Small ATT config",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/att_student_base.yaml"),
        help="Base ATT config",
    )
    parser.add_argument("--small-devices", default="0", help="CUDA devices for small ATT")
    parser.add_argument("--base-devices", default="0", help="CUDA devices for base ATT")
    parser.add_argument(
        "--parallel-tracks",
        action="store_true",
        help="Run both tracks concurrently; requires disjoint device pools",
    )
    parser.add_argument("--small-port", type=int, default=29503)
    parser.add_argument("--base-port", type=int, default=29504)
    parser.add_argument(
        "--small-output",
        type=Path,
        default=Path("outputs/att_student_small"),
        help="Output directory for small ATT",
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path("outputs/att_student_base"),
        help="Output directory for base ATT",
    )
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=48,
        help="Effective record batch preserved across GPU topologies",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to resume checkpoint or 'auto' to resume from latest checkpoint",
    )
    parser.add_argument(
        "--small-resume",
        type=str,
        default=None,
        help="Path to resume checkpoint for small ATT track or 'auto'",
    )
    parser.add_argument(
        "--base-resume",
        type=str,
        default=None,
        help="Path to resume checkpoint for base ATT track or 'auto'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the trainer's bounded diagnostic path",
    )
    args = parser.parse_args()

    tracks = ["small", "base"] if args.track == "both" else [args.track]
    devices = {"small": args.small_devices, "base": args.base_devices}
    if args.parallel_tracks and len(tracks) > 1:
        overlap = set(_device_ids(devices["small"])) & set(_device_ids(devices["base"]))
        if overlap:
            raise ValueError(f"Parallel tracks require disjoint CUDA devices; overlap: {overlap}")
        if args.small_port == args.base_port:
            raise ValueError("Parallel tracks require distinct rendezvous ports")

    outputs = {"small": args.small_output, "base": args.base_output}
    checkpoints = {"small": args.small_checkpoint, "base": args.base_checkpoint}
    configs = {"small": args.small_config, "base": args.base_config}
    ports = {"small": args.small_port, "base": args.base_port}
    if len(tracks) > 1 and outputs["small"].resolve() == outputs["base"].resolve():
        raise ValueError("ATT tracks require distinct output directories")

    def start(track: str) -> subprocess.Popen:
        resume_val = (
            args.small_resume
            if track == "small" and args.small_resume is not None
            else args.base_resume
            if track == "base" and args.base_resume is not None
            else args.resume
        )
        resume_path = None
        if resume_val == "auto":
            resume_path = find_latest_checkpoint(outputs[track])
            if resume_path:
                print(f"[{track.upper()}] Auto-resuming ATT from latest checkpoint: {resume_path}")
        elif resume_val:
            resume_path = Path(resume_val)

        return launch_track(
            track,
            checkpoint=checkpoints[track],
            manifest=args.train_manifest,
            config=configs[track],
            output_dir=outputs[track],
            devices=devices[track],
            port=ports[track],
            num_candidates=args.num_candidates,
            epochs=args.epochs,
            effective_batch_size=args.effective_batch_size,
            resume=resume_path,
            dry_run=args.dry_run,
        )

    failed = False
    if args.parallel_tracks:
        pending = [(track, start(track)) for track in tracks]
    else:
        pending = []
        for track in tracks:
            process = start(track)
            returncode = process.wait()
            process.log_handle.close()  # type: ignore[attr-defined]
            print(f"[{track.upper()}] Finished with exit code {returncode}")
            if returncode != 0:
                failed = True
                break

    for track, process in pending:
        returncode = process.wait()
        process.log_handle.close()  # type: ignore[attr-defined]
        print(f"[{track.upper()}] Finished with exit code {returncode}")
        failed = failed or returncode != 0

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

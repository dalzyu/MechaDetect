#!/usr/bin/env python3
"""Launch helper for Adversarial Transformation Training (ATT) tracks.

Orchestrates independent, concurrent execution of small (ViT-S) and base (ViT-B)
ATT tracks on disjoint 2-GPU pools to ensure strict hardware and network isolation:
- Track small: GPU 0, 1; master port 29502; output outputs/att_student_small
- Track base:  GPU 2, 3; master port 29503; output outputs/att_student_base

Usage:
  # Launch both tracks concurrently:
  python scripts/launch_att_tracks.py --track both \
      --small-checkpoint outputs/student_dinov3_small/checkpoint-promoted.pt \
      --base-checkpoint outputs/student_dinov3_base/checkpoint-promoted.pt

  # Launch small track only:
  python scripts/launch_att_tracks.py --track small \
      --small-checkpoint outputs/student_dinov3_small/checkpoint-promoted.pt

  # Launch base track only:
  python scripts/launch_att_tracks.py --track base \
      --base-checkpoint outputs/student_dinov3_base/checkpoint-promoted.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_track_command(
    variant: str,
    *,
    checkpoint: Path | str | None,
    manifest: Path | str | None,
    config: Path | str | None,
    output_dir: Path | str,
    num_candidates: int = 3,
    epochs: int = 1,
    port: int = 29502,
    nproc_per_node: int = 2,
    dry_run: bool = False,
    python_exe: str = sys.executable,
) -> list[str]:
    """Build command to run train_att.py with torchrun on a 2-GPU pool."""
    script_path = Path(__file__).resolve().parent / "train_att.py"

    cmd = [
        python_exe,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_port={port}",
        str(script_path),
        "--variant",
        variant,
        "--output-dir",
        str(output_dir),
        "--num-candidates",
        str(num_candidates),
        "--epochs",
        str(epochs),
    ]
    if checkpoint:
        cmd.extend(["--student-checkpoint", str(checkpoint)])
    if manifest:
        cmd.extend(["--manifest", str(manifest)])
    if config:
        cmd.extend(["--config", str(config)])
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
    gpu_ids: str,
    port: int,
    num_candidates: int = 3,
    epochs: int = 1,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Launch one ATT track as a supervised subprocess with isolated GPUs and port."""
    nproc = len([g for g in gpu_ids.split(",") if g.strip()])
    cmd = build_track_command(
        variant,
        checkpoint=checkpoint,
        manifest=manifest,
        config=config,
        output_dir=output_dir,
        num_candidates=num_candidates,
        epochs=epochs,
        port=port,
        nproc_per_node=max(1, nproc),
        dry_run=dry_run,
    )

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["MASTER_PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / f"att_launch_{variant}.log"
    log_handle = open(log_file, "w", encoding="utf-8")

    print(f"[{variant.upper()}] Spawning ATT track:")
    print(f"  CUDA_VISIBLE_DEVICES: {gpu_ids}")
    print(f"  MASTER_PORT:          {port}")
    print(f"  Output Directory:     {output_dir}")
    print(f"  Log File:             {log_file}")
    print(f"  Command:              {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    # Stash log handle on proc so caller can close it on exit
    proc.log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch isolated small/base ATT tracks")
    parser.add_argument(
        "--track",
        choices=["small", "base", "both"],
        default="both",
        help="Which track(s) to launch",
    )
    parser.add_argument(
        "--small-checkpoint",
        type=Path,
        default=None,
        help="Path to promoted float ViT-S checkpoint",
    )
    parser.add_argument(
        "--base-checkpoint", type=Path, default=None, help="Path to promoted float ViT-B checkpoint"
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
    parser.add_argument(
        "--small-gpus", type=str, default="0,1", help="CUDA_VISIBLE_DEVICES for small track"
    )
    parser.add_argument(
        "--base-gpus", type=str, default="2,3", help="CUDA_VISIBLE_DEVICES for base track"
    )
    parser.add_argument(
        "--small-port", type=int, default=29502, help="Rendezvous port for small track"
    )
    parser.add_argument(
        "--base-port", type=int, default=29503, help="Rendezvous port for base track"
    )
    parser.add_argument(
        "--small-output",
        type=Path,
        default=Path("outputs/att_student_small"),
        help="Output dir for small track",
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=Path("outputs/att_student_base"),
        help="Output dir for base track",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=3, help="Number of transform candidates per row"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of coverage passes (default: 1)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Launch 2-step dry run sanity check")

    args = parser.parse_args()

    # Track isolation check: verify GPU sets and ports are disjoint when running both
    if args.track == "both":
        small_gpu_set = set(args.small_gpus.split(","))
        base_gpu_set = set(args.base_gpus.split(","))
        overlap = small_gpu_set & base_gpu_set
        if overlap:
            raise ValueError(
                f"Track GPU collision! Small GPUs {args.small_gpus} and Base GPUs {args.base_gpus} share devices: {overlap}"
            )
        if args.small_port == args.base_port:
            raise ValueError(
                f"Track port collision! Small and Base cannot share port {args.small_port}"
            )
        if args.small_output.resolve() == args.base_output.resolve():
            raise ValueError(
                f"Track output collision! Small and Base cannot share output dir {args.small_output}"
            )

    procs: dict[str, subprocess.Popen] = {}

    try:
        if args.track in {"small", "both"}:
            proc_sm = launch_track(
                "small",
                checkpoint=args.small_checkpoint,
                manifest=args.train_manifest,
                config=args.small_config,
                output_dir=args.small_output,
                gpu_ids=args.small_gpus,
                port=args.small_port,
                num_candidates=args.num_candidates,
                epochs=args.epochs,
                dry_run=args.dry_run,
            )
            procs["small"] = proc_sm

        if args.track in {"base", "both"}:
            proc_ba = launch_track(
                "base",
                checkpoint=args.base_checkpoint,
                manifest=args.train_manifest,
                config=args.base_config,
                output_dir=args.base_output,
                gpu_ids=args.base_gpus,
                port=args.base_port,
                num_candidates=args.num_candidates,
                epochs=args.epochs,
                dry_run=args.dry_run,
            )
            procs["base"] = proc_ba

        # Wait for processes
        all_success = True
        for name, proc in procs.items():
            ret = proc.wait()
            if hasattr(proc, "log_handle"):
                proc.log_handle.close()
            if ret != 0:
                print(f"[{name.upper()}] ATT failed with exit code {ret}", file=sys.stderr)
                all_success = False
            else:
                print(f"[{name.upper()}] ATT completed successfully.")

        sys.exit(0 if all_success else 1)

    except KeyboardInterrupt:
        print("\nInterrupt received. Terminating ATT tracks...")
        for proc in procs.values():
            proc.terminate()
            if hasattr(proc, "log_handle"):
                proc.log_handle.close()
        sys.exit(130)


if __name__ == "__main__":
    main()

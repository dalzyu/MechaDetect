#!/usr/bin/env python3
"""Launch one or both DINOv3 student-distillation tracks.

The default topology is one GPU (device 0). When both tracks are requested they
run sequentially, so a workstation never loads both models at once. Advanced
multi-GPU runs are explicit through ``--small-devices`` / ``--base-devices``;
``--parallel-tracks`` additionally requires disjoint device pools.

Every topology preserves an effective record batch of 48 by scaling gradient
accumulation from each track's memory-safe physical batch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

TRACK_SPECS: dict[str, dict[str, Any]] = {
    "small": {
        "variant": "small",
        "port": 29501,
        "physical_batch_size": 12,
        "default_output_dir": "outputs/student_dinov3_small",
        "default_config": "configs/student_dinov3_small_distill.yaml",
        "exact_parameter_count": 25089666,
        "description": "DINOv3 ViT-S complete detector (25.1M params)",
    },
    "base": {
        "variant": "base",
        "port": 29502,
        "physical_batch_size": 3,
        "default_output_dir": "outputs/student_dinov3_base",
        "default_config": "configs/student_dinov3_base_distill.yaml",
        "exact_parameter_count": 89350914,
        "description": "DINOv3 ViT-B complete detector (89.4M params)",
    },
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
    teacher_config: Path | str,
    teacher_checkpoint: Path | str,
    manifest: Path | str,
    output_dir: Path | str,
    val_manifest: Path | str | None = None,
    teacher_promotion_report: Path | str | None = None,
    student_config: Path | str | None = None,
    epochs: int = 2,
    seed: int = 42,
    world_size: int = 1,
    physical_batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    num_workers: int = 4,
    resume: Path | str | None = None,
    dry_run: bool = False,
    skip_teacher_gate: bool = False,
    python_exe: str = sys.executable,
) -> list[str]:
    """Build one direct or DDP student-training command."""
    script_path = Path(__file__).resolve().parent / "distill_student.py"
    spec = TRACK_SPECS[variant]
    physical = physical_batch_size or int(spec["physical_batch_size"])
    accumulation = gradient_accumulation or 48 // (physical * world_size)

    if world_size == 1:
        cmd = [python_exe, str(script_path)]
    else:
        cmd = [
            python_exe,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            f"--master_port={spec['port']}",
            str(script_path),
        ]
    cmd.extend(
        [
            "--student",
            variant,
            "--teacher-config",
            str(teacher_config),
            "--teacher-checkpoint",
            str(teacher_checkpoint),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--world-size",
            str(world_size),
            "--physical-batch-size",
            str(physical),
            "--gradient-accumulation",
            str(accumulation),
            "--num-workers",
            str(num_workers if sys.platform != "win32" else 0),
        ]
    )
    if dry_run:
        cmd.append("--dry-run")
    if skip_teacher_gate:
        cmd.append("--skip-teacher-gate")
    if val_manifest:
        cmd.extend(["--val-manifest", str(val_manifest)])
    if teacher_promotion_report:
        cmd.extend(["--teacher-promotion-report", str(teacher_promotion_report)])
    if student_config:
        cmd.extend(["--student-config", str(student_config)])
    if resume:
        cmd.extend(["--resume", str(resume)])
    return cmd


def launch_track(
    variant: str,
    *,
    teacher_config: Path | str,
    teacher_checkpoint: Path | str,
    manifest: Path | str,
    output_dir: Path | str,
    devices: str = "0",
    val_manifest: Path | str | None = None,
    teacher_promotion_report: Path | str | None = None,
    student_config: Path | str | None = None,
    epochs: int = 2,
    seed: int = 42,
    effective_batch_size: int = 48,
    num_workers: int = 4,
    resume: Path | str | None = None,
    dry_run: bool = False,
    skip_teacher_gate: bool = False,
) -> subprocess.Popen:
    """Launch one student track on its explicit CUDA device pool."""
    spec = TRACK_SPECS[variant]
    world_size, physical_batch_size, gradient_accumulation = _batch_geometry(
        variant, devices, effective_batch_size
    )
    cmd = build_track_command(
        variant,
        teacher_config=teacher_config,
        teacher_checkpoint=teacher_checkpoint,
        manifest=manifest,
        output_dir=output_dir,
        val_manifest=val_manifest,
        teacher_promotion_report=teacher_promotion_report,
        student_config=student_config,
        epochs=epochs,
        seed=seed,
        world_size=world_size,
        physical_batch_size=physical_batch_size,
        gradient_accumulation=gradient_accumulation,
        num_workers=num_workers,
        resume=resume,
        dry_run=dry_run,
        skip_teacher_gate=skip_teacher_gate,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["MASTER_PORT"] = str(spec["port"])
    env["PYTHONUNBUFFERED"] = "1"

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    log_file = out_p / f"distill_{variant}.log"
    log_fp = open(log_file, "w", encoding="utf-8")

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
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc.log_handle = log_fp  # type: ignore[attr-defined]
    return proc


def select_default_student_model(
    small_report_path: Path | str,
    base_report_path: Path | str,
    output_dir: Path | str = Path("outputs"),
) -> dict[str, Any]:
    """Evaluate small vs base float student tracks and make the default model selection decision.

    Decision rule:
      ViT-S (small, 25.1M complete detector) remains default edge artifact unless:
      1. ViT-S failed class-recall gate (recall < 0.82) AND ViT-B fixes it (recalls >= 0.82), OR
      2. ViT-B gains at least 0.01 (1pp) in worst-transformation AUROC over ViT-S, subject to ViT-B passing all gates.
    """
    small_p = Path(small_report_path)
    base_p = Path(base_report_path)
    if not small_p.is_file():
        raise FileNotFoundError(f"Small student promotion report not found: {small_p}")
    if not base_p.is_file():
        raise FileNotFoundError(f"Base student promotion report not found: {base_p}")

    small_rep = json.loads(small_p.read_text(encoding="utf-8"))
    base_rep = json.loads(base_p.read_text(encoding="utf-8"))

    small_metrics = small_rep.get("metrics", {})
    base_metrics = base_rep.get("metrics", {})

    small_passed = bool(small_rep.get("passed", False))
    base_passed = bool(base_rep.get("passed", False))

    small_clean = float(small_metrics.get("clean", {}).get("ai_positive_auroc", 0.0))
    small_worst = float(small_metrics.get("worst", {}).get("worst_auroc", 0.0))
    small_pos_rec = float(small_metrics.get("clean", {}).get("ai_positive_recall", 0.0))
    small_auth_rec = float(small_metrics.get("clean", {}).get("authentic_recall", 0.0))

    base_clean = float(base_metrics.get("clean", {}).get("ai_positive_auroc", 0.0))
    base_worst = float(base_metrics.get("worst", {}).get("worst_auroc", 0.0))
    base_pos_rec = float(base_metrics.get("clean", {}).get("ai_positive_recall", 0.0))
    base_auth_rec = float(base_metrics.get("clean", {}).get("authentic_recall", 0.0))

    worst_gain = base_worst - small_worst
    clean_gain = base_clean - small_clean

    small_recall_failure = small_pos_rec < 0.82 or small_auth_rec < 0.82
    base_recall_passed = base_pos_rec >= 0.82 and base_auth_rec >= 0.82

    select_base = False
    decision_reason = ""

    if small_recall_failure and base_recall_passed and base_passed:
        select_base = True
        decision_reason = (
            f"ViT-B selected: ViT-S failed recall gate (pos={small_pos_rec:.4f}, auth={small_auth_rec:.4f}) "
            f"which ViT-B resolves (pos={base_pos_rec:.4f}, auth={base_auth_rec:.4f})"
        )
    elif base_passed and worst_gain >= 0.01:
        select_base = True
        decision_reason = (
            f"ViT-B selected: worst-transform AUROC gain {worst_gain:.4f} exceeds 1.0pp threshold "
            f"(base={base_worst:.4f}, small={small_worst:.4f})"
        )
    else:
        select_base = False
        decision_reason = (
            f"ViT-S selected as default edge model (25.1M complete detector). "
            f"ViT-B worst-transform gain {worst_gain:.4f} does not justify extra compute (threshold >= 0.01)"
        )

    selected_track = "base" if select_base else "small"

    decision_report = {
        "selected_track": selected_track,
        "selected_parameter_count": TRACK_SPECS[selected_track]["exact_parameter_count"],
        "decision_reason": decision_reason,
        "comparison": {
            "worst_transform_auroc_gain": worst_gain,
            "clean_auroc_gain": clean_gain,
            "vit_s": {
                "checkpoint": small_rep.get("checkpoint_path"),
                "passed": small_passed,
                "clean_auroc": small_clean,
                "worst_auroc": small_worst,
                "ai_positive_recall": small_pos_rec,
                "authentic_recall": small_auth_rec,
            },
            "vit_b": {
                "checkpoint": base_rep.get("checkpoint_path"),
                "passed": base_passed,
                "clean_auroc": base_clean,
                "worst_auroc": base_worst,
                "ai_positive_recall": base_pos_rec,
                "authentic_recall": base_auth_rec,
            },
        },
    }

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    report_file = out_p / "student_selection_report.json"
    report_file.write_text(json.dumps(decision_report, indent=2), encoding="utf-8")
    print(f"\nDefault student model selected: {selected_track.upper()} ({decision_reason})")
    print(f"Selection report written to: {report_file}")
    return decision_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch single-GPU student distillation by default; multi-GPU is explicit."
    )
    parser.add_argument(
        "--track",
        choices=["small", "base", "both"],
        default="both",
        help="Student track(s) to run sequentially by default",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        required=True,
        help="Path to final teacher checkpoint (.pt)",
    )
    parser.add_argument(
        "--teacher-config",
        type=Path,
        default=Path("configs/teacher_dinov3_stage2_paired_unfrozen.yaml"),
        help="Path to teacher configuration YAML",
    )
    parser.add_argument(
        "--teacher-promotion-report",
        type=Path,
        default=None,
        help="Path to teacher promotion_report.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("splits/production_eligible/train.parquet"),
        help="Path to eligible training manifest parquet",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("splits/production_eligible/validation.parquet"),
        help="Path to validation manifest parquet",
    )
    parser.add_argument(
        "--small-output-dir",
        type=Path,
        default=Path("outputs/student_dinov3_small"),
        help="Output directory for the small student",
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("outputs/student_dinov3_base"),
        help="Output directory for the base student",
    )
    parser.add_argument(
        "--small-config",
        type=Path,
        default=Path("configs/student_dinov3_small_distill.yaml"),
        help="Configuration YAML for the small student",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/student_dinov3_base_distill.yaml"),
        help="Configuration YAML for the base student",
    )
    parser.add_argument("--small-devices", default="0", help="CUDA devices for the small track")
    parser.add_argument("--base-devices", default="0", help="CUDA devices for the base track")
    parser.add_argument(
        "--parallel-tracks",
        action="store_true",
        help="Run both tracks concurrently; requires disjoint device pools",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=48,
        help="Effective record batch preserved across GPU topologies",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of complete deterministic passes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-teacher-gate",
        action="store_true",
        help="Use a final teacher checkpoint without promotion-report verification",
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
        help="Path to resume checkpoint for small student or 'auto'",
    )
    parser.add_argument(
        "--base-resume",
        type=str,
        default=None,
        help="Path to resume checkpoint for base student or 'auto'",
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

    outputs = {
        "small": args.small_output_dir,
        "base": args.base_output_dir,
    }
    configs = {"small": args.small_config, "base": args.base_config}
    if len(tracks) > 1 and outputs["small"].resolve() == outputs["base"].resolve():
        raise ValueError("Student tracks require distinct output directories")

    print("=" * 65)
    print("MechaDetect Student Distillation Launcher")
    print(f"Tracks:          {tracks}")
    print(f"Execution:       {'parallel' if args.parallel_tracks else 'sequential'}")
    print(f"Device pools:    {devices}")
    print(f"Effective batch: {args.effective_batch_size}")
    print("=" * 65)

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
                print(f"[{track.upper()}] Auto-resuming from latest checkpoint: {resume_path}")
        elif resume_val:
            resume_path = Path(resume_val)

        return launch_track(
            track,
            teacher_config=args.teacher_config,
            teacher_checkpoint=args.teacher_checkpoint,
            manifest=args.manifest,
            output_dir=outputs[track],
            devices=devices[track],
            val_manifest=args.val_manifest,
            teacher_promotion_report=args.teacher_promotion_report,
            student_config=configs[track] if configs[track].is_file() else None,
            epochs=args.epochs,
            seed=args.seed,
            effective_batch_size=args.effective_batch_size,
            num_workers=args.num_workers,
            resume=resume_path,
            dry_run=args.dry_run,
            skip_teacher_gate=args.skip_teacher_gate,
        )

    processes: dict[str, subprocess.Popen] = {}
    failed = False
    if args.parallel_tracks:
        processes = {track: start(track) for track in tracks}
        pending = list(processes.items())
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

    if failed:
        raise SystemExit(1)
    if args.track == "both" and not args.dry_run:
        small_report = args.small_output_dir / "promotion_report.json"
        base_report = args.base_output_dir / "promotion_report.json"
        if small_report.is_file() and base_report.is_file():
            select_default_student_model(
                small_report, base_report, output_dir=args.small_output_dir.parent
            )
    print("All requested student distillation operations completed.")


if __name__ == "__main__":
    main()

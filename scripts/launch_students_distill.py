#!/usr/bin/env python3
"""Launch helper for independent DINOv3 ViT-S and ViT-B student distillation tracks.

Orchestrates isolated 2-GPU execution of small and base student distillation tracks
on disjoint GPU pools via torch.distributed.run to ensure strict hardware and network isolation:
- Small Track (ViT-S): GPUs 0, 1; master port 29501; output outputs/student_dinov3_small
- Base Track (ViT-B):  GPUs 2, 3; master port 29502; output outputs/student_dinov3_base

Both tracks share:
- Same promoted teacher checkpoint hash and promotion gate verification
- Same immutable training manifest and train-only membership
- Same deterministic coverage sampler and 2 complete passes
- Same effective record batch (48) and random seed (42)
- Independent parameter counting, coverage milestones, and external validation promotion
- Default-model selection decision: ViT-S unless ViT-B gains >= 1.0pp worst-transform AUROC or fixes ViT-S recall failure

Usage:
  # Launch both tracks concurrently:
  python scripts/launch_students_distill.py --track both \
      --teacher-checkpoint outputs/teacher_stage2/checkpoint-promoted.pt \
      --teacher-promotion-report outputs/teacher_stage2/promotion_report.json

  # Launch small track only:
  python scripts/launch_students_distill.py --track small \
      --teacher-checkpoint outputs/teacher_stage2/checkpoint-promoted.pt

  # Launch base track only:
  python scripts/launch_students_distill.py --track base \
      --teacher-checkpoint outputs/teacher_stage2/checkpoint-promoted.pt
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
        "devices": "0,1",
        "port": 29501,
        "default_output_dir": "outputs/student_dinov3_small",
        "default_config": "configs/student_dinov3_small_distill.yaml",
        "exact_parameter_count": 25089666,
        "description": "DINOv3 ViT-S complete detector (25.1M params)",
    },
    "base": {
        "variant": "base",
        "devices": "2,3",
        "port": 29502,
        "default_output_dir": "outputs/student_dinov3_base",
        "default_config": "configs/student_dinov3_base_distill.yaml",
        "exact_parameter_count": 89350914,
        "description": "DINOv3 ViT-B complete detector (89.4M params)",
    },
}


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
    world_size: int = 2,
    physical_batch_size: int = 12,
    gradient_accumulation: int = 2,
    num_workers: int = 4,
    dry_run: bool = False,
    python_exe: str = sys.executable,
) -> list[str]:
    """Build the command invocation for a single student track using torch.distributed.run."""
    script_path = Path(__file__).resolve().parent / "distill_student.py"
    spec = TRACK_SPECS[variant]

    cmd = [
        python_exe,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        f"--master_port={spec['port']}",
        str(script_path),
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
        str(physical_batch_size),
        "--gradient-accumulation",
        str(gradient_accumulation),
        "--num-workers",
        str(num_workers if sys.platform != "win32" else 0),
    ]
    if dry_run:
        cmd.append("--dry-run")

    if val_manifest:
        cmd.extend(["--val-manifest", str(val_manifest)])
    if teacher_promotion_report:
        cmd.extend(["--teacher-promotion-report", str(teacher_promotion_report)])
    if student_config:
        cmd.extend(["--student-config", str(student_config)])

    return cmd


def launch_track(
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
    world_size: int = 2,
    physical_batch_size: int = 12,
    gradient_accumulation: int = 2,
    dry_run: bool = False,
) -> subprocess.Popen:
    """Launch a student track as a background process with isolated GPU/port environment."""
    spec = TRACK_SPECS[variant]
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
        dry_run=dry_run,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(spec["devices"])
    env["MASTER_PORT"] = str(spec["port"])
    env["PYTHONUNBUFFERED"] = "1"

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    log_file = out_p / f"distill_{variant}.log"
    log_fp = open(log_file, "w", encoding="utf-8")

    print(f"[{variant.upper()}] Launching on GPUs {spec['devices']} (port {spec['port']})...")
    print(f"[{variant.upper()}] Log file: {log_file}")
    print(f"[{variant.upper()}] Command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
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
        "selected_model": selected_track,
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
        description="Launch isolated 2-GPU DINOv3 student distillation tracks (small and/or base)."
    )
    parser.add_argument(
        "--track",
        choices=["small", "base", "both"],
        default="both",
        help="Which student track to launch (default: both concurrently)",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        required=True,
        help="Path to promoted teacher checkpoint (.pt)",
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
        help="Output directory for small student",
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("outputs/student_dinov3_base"),
        help="Output directory for base student",
    )
    parser.add_argument(
        "--small-config",
        type=Path,
        default=Path("configs/student_dinov3_small_distill.yaml"),
        help="Configuration YAML for small student",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/student_dinov3_base_distill.yaml"),
        help="Configuration YAML for base student",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of complete deterministic passes (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed shared across tracks (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify isolation and arguments without training",
    )
    parser.add_argument(
        "--strict-teacher-gate",
        action="store_true",
        default=True,
        help="Always strictly enforce teacher promotion gate (always True, cannot be bypassed)",
    )

    args = parser.parse_args()

    tracks_to_run = ["small", "base"] if args.track == "both" else [args.track]

    print("=" * 65)
    print("MechaDetect 4x4090 Student Distillation Launcher")
    print(f"Tracks selected: {tracks_to_run}")
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"Training manifest:  {args.manifest}")
    print(f"Validation manifest: {args.val_manifest}")
    print(f"Pass budget:        {args.epochs} complete passes")
    print("Effective batch:    48 (12 physical x 2 GPUs x 2 accum)")
    print("=" * 65)

    # Validate isolation constraints before launch
    ports = [TRACK_SPECS[t]["port"] for t in tracks_to_run]
    assert len(ports) == len(set(ports)), f"Port collision detected: {ports}"

    outputs = [
        args.small_output_dir if t == "small" else args.base_output_dir for t in tracks_to_run
    ]
    assert len(outputs) == len(set(outputs)), f"Output directory collision detected: {outputs}"

    processes: dict[str, subprocess.Popen] = {}
    for track in tracks_to_run:
        out_dir = args.small_output_dir if track == "small" else args.base_output_dir
        cfg = args.small_config if track == "small" else args.base_config
        proc = launch_track(
            track,
            teacher_config=args.teacher_config,
            teacher_checkpoint=args.teacher_checkpoint,
            manifest=args.manifest,
            output_dir=out_dir,
            val_manifest=args.val_manifest,
            teacher_promotion_report=args.teacher_promotion_report,
            student_config=cfg if cfg.is_file() else None,
            epochs=args.epochs,
            seed=args.seed,
            world_size=2,
            physical_batch_size=12,
            gradient_accumulation=2,
            dry_run=args.dry_run,
        )
        processes[track] = proc

    print(
        f"\nAll requested tracks launched ({len(processes)} process(es)). Waiting for completion..."
    )

    failed = False
    for track, proc in processes.items():
        returncode = proc.wait()
        status_str = "SUCCESS" if returncode == 0 else f"FAILED (code {returncode})"
        print(f"[{track.upper()}] Finished with status: {status_str}")
        if returncode != 0:
            failed = True

    if failed:
        print("\nOne or more student distillation tracks failed. Inspect log files for details.")
        sys.exit(1)

    # Run default model selection when both tracks have completed
    if args.track == "both" and not args.dry_run:
        small_rep_path = args.small_output_dir / "promotion_report.json"
        base_rep_path = args.base_output_dir / "promotion_report.json"
        if small_rep_path.is_file() and base_rep_path.is_file():
            select_default_student_model(
                small_rep_path, base_rep_path, output_dir=args.small_output_dir.parent
            )

    print("\nAll requested student distillation operations completed.")
    sys.exit(0)


if __name__ == "__main__":
    main()

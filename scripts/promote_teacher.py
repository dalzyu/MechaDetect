#!/usr/bin/env python3
"""External Teacher Promotion & Checkpoint Selection Script.

Evaluates and ranks teacher checkpoints (25%, 50%, 75%, 100% coverage or steps)
outside DDP on frozen validation IDs and the complete single-transform severity grid.

Selects the best checkpoint according to deterministic promotion criteria:
  1. Gate satisfaction (Clean AUROC > 0.96, both recalls >= 0.82 at calibrated threshold)
  2. Highest clean AUROC
  3. Highest worst-transformation AUROC
  4. Highest mean-transformation AUROC
  5. Highest balanced accuracy at calibrated threshold
  6. Deterministic path tie-breaker

Emits the production promotion report (promotion_report.json) and metadata sidecar (metadata.json).
Never evaluates, selects, or calibrates on test/test_unseen/organizer demo data.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

try:
    from scripts.evaluate_teacher import (
        create_metadata_sidecar,
        evaluate_teacher_checkpoint,
    )
except ImportError:
    from evaluate_teacher import (
        create_metadata_sidecar,
        evaluate_teacher_checkpoint,
    )


def rank_checkpoints(candidate_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank evaluated candidate checkpoints deterministically.

    Ranking Order:
      1. Gate satisfaction (passed == True first)
      2. Clean AUROC (descending)
      3. Worst-transformation AUROC (descending)
      4. Mean-transformation AUROC (descending)
      5. Balanced accuracy at calibrated threshold (descending)
      6. Checkpoint path (ascending string for reproducible tie-breaking)
    """

    def sort_key(item: dict[str, Any]) -> tuple[int, float, float, float, float, str]:
        report = item["report"]
        passed = bool(report.get("passed", False))
        metrics = report.get("metrics", {})

        clean_auroc = float(metrics.get("clean", {}).get("ai_positive_auroc") or 0.0)
        worst_auroc = float(metrics.get("worst", {}).get("worst_auroc") or 0.0)
        mean_auroc = float(metrics.get("mean", {}).get("mean_auroc") or 0.0)
        calibrated_bal_acc = float(
            metrics.get("calibrated", {}).get("clean", {}).get("balanced_accuracy") or 0.0
        )
        ckpt_path = str(report.get("checkpoint_path", ""))

        # Python sorts ascending: we use 0 for passed, 1 for failed
        return (
            0 if passed else 1,
            -clean_auroc,
            -worst_auroc,
            -mean_auroc,
            -calibrated_bal_acc,
            ckpt_path,
        )

    return sorted(candidate_results, key=sort_key)


def promote_teacher_checkpoints(
    checkpoints: list[Path] | None = None,
    eval_reports: list[Path] | None = None,
    manifest_path: Path = Path("splits/production_eligible/validation.parquet"),
    config_path: Path = Path("configs/teacher_dinov3_stage2_paired_unfrozen.yaml"),
    data_root: Path | None = None,
    batch_size: int = 4,
    device: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate and promote the best teacher checkpoint from candidates."""
    evaluated_candidates: list[dict[str, Any]] = []

    # Case A: Load pre-evaluated reports if provided
    if eval_reports:
        for rep_path in eval_reports:
            p = Path(rep_path)
            if not p.is_file():
                raise FileNotFoundError(f"Evaluation report not found: {p}")
            data = json.loads(p.read_text(encoding="utf-8"))
            if "report" in data:
                evaluated_candidates.append(data)
            elif "checkpoint_path" in data and "metrics" in data:
                # Bare report format
                sidecar = create_metadata_sidecar(
                    manifest_digest_val=data["manifest_digest"],
                    calibrated_threshold=data["calibrated_threshold"],
                    passed=data["passed"],
                    parameter_count=data.get("parameter_count"),
                )
                evaluated_candidates.append({"report": data, "metadata": sidecar})
            else:
                raise ValueError(f"Unrecognized evaluation report structure in {p}")

    # Case B: Evaluate checkpoints that were not pre-evaluated
    if checkpoints:
        for ckpt in checkpoints:
            ckpt_path = Path(ckpt)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
            print(f"\nEvaluating candidate checkpoint: {ckpt_path}...")
            res = evaluate_teacher_checkpoint(
                checkpoint_path=ckpt_path,
                manifest_path=manifest_path,
                config_path=config_path,
                data_root=data_root,
                batch_size=batch_size,
                device=device,
                limit=limit,
            )
            evaluated_candidates.append(res)

    if not evaluated_candidates:
        raise ValueError("No candidate checkpoints or evaluation reports provided for promotion.")

    # Deterministic ranking
    ranked = rank_checkpoints(evaluated_candidates)
    winner = ranked[0]
    winner_report = winner["report"]
    winner_metadata = winner["metadata"]
    winner_passed = bool(winner_report.get("passed", False))

    return {
        "selected_candidate": winner_report["checkpoint_path"],
        "promoted_checkpoint": winner_report["checkpoint_path"] if winner_passed else None,
        "promotion_report": winner_report,
        "metadata_sidecar": winner_metadata,
        "ranked_candidates": ranked,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and promote the best MechaDetect teacher checkpoint."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="One or more checkpoint paths or globs (e.g. outputs/teacher_stage2/checkpoint-*.pt)",
    )
    parser.add_argument(
        "--eval-reports",
        nargs="+",
        help="One or more precomputed evaluation JSON report files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("splits/production_eligible/validation.parquet"),
        help="Path to frozen validation manifest (Parquet, JSONL, or CSV)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher_dinov3_stage2_paired_unfrozen.yaml"),
        help="Path to teacher training/model config YAML",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data root directory containing validation images",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("outputs/teacher_stage2/promotion_report.json"),
        help="Path to write final promotion_report.json",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("outputs/teacher_stage2/metadata.json"),
        help="Path to write final metadata.json sidecar",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Optional path to copy the promoted checkpoint if gate criteria are met.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick testing",
    )
    return parser.parse_args()


def expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        matched = glob.glob(pat)
        if matched:
            paths.extend(Path(m) for m in sorted(matched))
        else:
            paths.append(Path(pat))
    return paths


def main() -> None:
    args = parse_args()

    checkpoint_paths: list[Path] = []
    if args.checkpoints:
        checkpoint_paths = expand_globs(args.checkpoints)

    eval_report_paths: list[Path] = []
    if args.eval_reports:
        eval_report_paths = expand_globs(args.eval_reports)

    if not checkpoint_paths and not eval_report_paths:
        print("Error: Must provide either --checkpoints or --eval-reports", file=sys.stderr)
        sys.exit(1)

    print("=== MechaDetect Teacher Promotion Pipeline ===")
    if checkpoint_paths:
        print(
            f"Candidate Checkpoints ({len(checkpoint_paths)}): {[str(p) for p in checkpoint_paths]}"
        )
    if eval_report_paths:
        print(
            f"Pre-evaluated Reports ({len(eval_report_paths)}): {[str(p) for p in eval_report_paths]}"
        )
    print(f"Validation Manifest: {args.manifest}")

    result = promote_teacher_checkpoints(
        checkpoints=checkpoint_paths,
        eval_reports=eval_report_paths,
        manifest_path=args.manifest,
        config_path=args.config,
        data_root=args.data_root,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    report = result["promotion_report"]
    sidecar = result["metadata_sidecar"]
    ranked = result["ranked_candidates"]

    # Display Candidates Table
    print("\n--- Candidate Rankings ---")
    header = f"{'Rank':<5} {'Gate':<8} {'Clean AUROC':<13} {'Worst AUROC':<13} {'Mean AUROC':<12} {'BalAcc @ Cal':<14} {'Checkpoint'}"
    print(header)
    print("-" * len(header))

    for idx, cand in enumerate(ranked, start=1):
        rep = cand["report"]
        p_status = "PASS" if rep["passed"] else "FAIL"
        c_auroc = (
            f"{rep['metrics']['clean']['ai_positive_auroc']:.4f}"
            if rep["metrics"]["clean"]["ai_positive_auroc"]
            else "N/A"
        )
        w_auroc = (
            f"{rep['metrics']['worst']['worst_auroc']:.4f}"
            if rep["metrics"]["worst"]["worst_auroc"]
            else "N/A"
        )
        m_auroc = (
            f"{rep['metrics']['mean']['mean_auroc']:.4f}"
            if rep["metrics"]["mean"]["mean_auroc"]
            else "N/A"
        )
        bal_acc = (
            f"{rep['metrics']['calibrated']['clean']['balanced_accuracy']:.4f}"
            if rep["metrics"]["calibrated"]["clean"]["balanced_accuracy"]
            else "N/A"
        )
        ckpt_name = Path(rep["checkpoint_path"]).name
        print(
            f"{idx:<5} {p_status:<8} {c_auroc:<13} {w_auroc:<13} {m_auroc:<12} {bal_acc:<14} {ckpt_name}"
        )

    print("\n--- Promotion Result ---")
    if report["passed"]:
        print(f"Promoted Checkpoint:   {report['checkpoint_path']}")
        print("Promotion Status:      PROMOTED (Passed Gate)")
    else:
        print(
            f"Selected Candidate:    {report['checkpoint_path']} (Best candidate, did not pass gate)"
        )
        print("Promoted Checkpoint:   None")
        print("Promotion Status:      REJECTED (Failed Gate)")

    if not report["passed"]:
        print("Failure Reasons:")
        for r in report["failed_reasons"]:
            print(f"  - {r}")

    # Write output files
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPromotion report saved to: {args.output_report}")

    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Metadata sidecar saved to: {args.output_metadata}")

    if args.output_checkpoint is not None:
        if report["passed"]:
            import shutil

            args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(report["checkpoint_path"], args.output_checkpoint)
            print(f"Promoted checkpoint copied to: {args.output_checkpoint}")
        else:
            print(
                f"Gate check rejected: not copying checkpoint to {args.output_checkpoint}",
                file=sys.stderr,
            )
    # Exit code: 0 if passed, 1 if failed gate (enforcing stop before distillation)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""External comparison and promotion gate for Adversarial Transformation Training (ATT).

Evaluates post-ATT student metrics against pre-ATT float student baselines:
1. Worst-transformation AUROC improves (> float baseline).
2. Worst-domain/generator AUROC does not regress (>= float baseline - 0.1pp tolerance).
3. Clean AUROC loss is <= 0.5 percentage points (0.005).
4. Both AI-positive recall and authentic recall remain >= 0.82.
5. Emits structured JSON promotion report (track-level and shared multi-track).

Usage:
  # Single track check:
  python scripts/check_att_gate.py \
      --float-eval outputs/eval_float_small.json \
      --att-eval outputs/eval_att_small.json \
      --track small \
      --output outputs/att_small_gate_report.json

  # Shared dual-track promotion check:
  python scripts/check_att_gate.py \
      --small-float-eval outputs/eval_float_small.json \
      --small-att-eval outputs/eval_att_small.json \
      --base-float-eval outputs/eval_float_base.json \
      --base-att-eval outputs/eval_att_base.json \
      --shared-report outputs/att_shared_promotion_report.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

CLEAN_AUROC_MAX_DROP = 0.005  # <= 0.5 percentage points regression
RECALL_MIN_THRESHOLD = 0.82  # Recalls must be >= 0.82
DOMAIN_REGRESSION_TOLERANCE = 0.001  # Numerical tolerance (0.1pp) for nonregression


ATT_METADATA: dict[str, dict[str, Any]] = {
    "small": {
        "model_family": "dinov3-vits16",
        "parameter_count": 25089666,
        "image_size": 224,
    },
    "base": {
        "model_family": "dinov3-vitb16",
        "parameter_count": 89350914,
        "image_size": 224,
    },
}


def compute_file_sha256(path: Path | str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for SHA256 computation: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_copy_checkpoint(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"ATT candidate checkpoint not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_evaluation_contract(path: Path | None) -> tuple[dict[str, Any], str, float]:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"ATT evaluation report not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest_digest = str(data.get("manifest_digest", ""))
    if len(manifest_digest) != 64 or any(
        char not in "0123456789abcdef" for char in manifest_digest.lower()
    ):
        raise ValueError(f"ATT evaluation report has no valid manifest_digest: {path}")
    raw_threshold = data.get("calibrated_threshold", data.get("operating_threshold"))
    if raw_threshold is None:
        raise ValueError(f"ATT evaluation report has no calibrated operating threshold: {path}")
    threshold = float(raw_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"ATT evaluation threshold must be in [0, 1], got {threshold}")
    return data, manifest_digest, threshold


def extract_gate_metrics(eval_data: dict[str, Any]) -> dict[str, float | None]:
    """Extract key gate metrics from an evaluate_performance.py JSON payload."""
    conditions = eval_data.get("conditions", {})
    clean_cond = conditions.get("clean", {})
    selection = eval_data.get("selection_summary", {})

    clean_auroc = clean_cond.get("ai_positive_auroc")
    ai_recall = clean_cond.get("binary_recall")
    auth_recall = clean_cond.get("authentic_recall")

    # Worst transformed AUROC
    worst_trans = selection.get("worst_transformed_auroc")
    if worst_trans is None:
        trans_aurocs = [
            v.get("ai_positive_auroc")
            for k, v in conditions.items()
            if k != "clean" and isinstance(v, dict) and v.get("ai_positive_auroc") is not None
        ]
        worst_trans = min(trans_aurocs) if trans_aurocs else None

    # Worst domain/generator AUROC
    worst_domain = selection.get("worst_generator_auroc")
    if worst_domain is None:
        gen_data = eval_data.get("per_ai_generator", {})
        gen_aurocs = [
            v.get("ai_positive_auroc")
            for v in gen_data.values()
            if isinstance(v, dict) and v.get("ai_positive_auroc") is not None
        ]
        worst_domain = min(gen_aurocs) if gen_aurocs else None

    return {
        "clean_auroc": clean_auroc,
        "ai_positive_recall": ai_recall,
        "authentic_recall": auth_recall,
        "worst_transformed_auroc": worst_trans,
        "worst_domain_auroc": worst_domain,
    }


def evaluate_track_gate(
    float_metrics: dict[str, float | None],
    att_metrics: dict[str, float | None],
    track_name: str = "student",
) -> dict[str, Any]:
    """Evaluate ATT promotion gate criteria for one student track."""
    checks: dict[str, Any] = {}
    failure_reasons: list[str] = []

    # 1. Worst-transformation AUROC improves
    fl_wt = float_metrics.get("worst_transformed_auroc")
    att_wt = att_metrics.get("worst_transformed_auroc")
    if fl_wt is None or att_wt is None:
        passed_wt = False
        msg = f"[{track_name}] Worst-transform AUROC missing from evaluation"
        failure_reasons.append(msg)
    elif att_wt > fl_wt:
        passed_wt = True
        delta = att_wt - fl_wt
        msg = f"[{track_name}] Worst-transform AUROC improved: {att_wt:.4f} > float {fl_wt:.4f} (+{delta:.4f}) -> PASS"
    else:
        passed_wt = False
        delta = att_wt - fl_wt
        msg = f"[{track_name}] Worst-transform AUROC did not improve: {att_wt:.4f} <= float {fl_wt:.4f} ({delta:+.4f}) -> FAIL"
        failure_reasons.append(msg)

    checks["worst_transform_improves"] = {
        "passed": passed_wt,
        "pre_att": fl_wt,
        "post_att": att_wt,
        "delta": (att_wt - fl_wt) if fl_wt is not None and att_wt is not None else None,
        "message": msg,
    }

    # 2. Worst-domain AUROC nonregression
    fl_wd = float_metrics.get("worst_domain_auroc")
    att_wd = att_metrics.get("worst_domain_auroc")
    if fl_wd is None or att_wd is None:
        passed_wd = False
        msg = f"[{track_name}] Worst-domain AUROC missing from evaluation"
        failure_reasons.append(msg)
    elif att_wd >= (fl_wd - DOMAIN_REGRESSION_TOLERANCE):
        passed_wd = True
        delta = att_wd - fl_wd
        msg = f"[{track_name}] Worst-domain AUROC nonregressed: {att_wd:.4f} vs float {fl_wd:.4f} ({delta:+.4f}) -> PASS"
    else:
        passed_wd = False
        delta = att_wd - fl_wd
        msg = f"[{track_name}] Worst-domain AUROC regressed: {att_wd:.4f} < float {fl_wd:.4f} ({delta:+.4f}) -> FAIL"
        failure_reasons.append(msg)

    checks["worst_domain_nonregress"] = {
        "passed": passed_wd,
        "pre_att": fl_wd,
        "post_att": att_wd,
        "delta": (att_wd - fl_wd) if fl_wd is not None and att_wd is not None else None,
        "message": msg,
    }

    # 3. Clean AUROC loss <= 0.5pp
    fl_cl = float_metrics.get("clean_auroc")
    att_cl = att_metrics.get("clean_auroc")
    if fl_cl is None or att_cl is None:
        passed_cl = False
        msg = f"[{track_name}] Clean AUROC missing from evaluation"
        failure_reasons.append(msg)
    else:
        drop = fl_cl - att_cl
        if drop <= CLEAN_AUROC_MAX_DROP:
            passed_cl = True
            msg = f"[{track_name}] Clean AUROC drop within tolerance: {drop:.4f} <= {CLEAN_AUROC_MAX_DROP:.4f} (att {att_cl:.4f}, float {fl_cl:.4f}) -> PASS"
        else:
            passed_cl = False
            msg = f"[{track_name}] Clean AUROC drop exceeded tolerance: {drop:.4f} > {CLEAN_AUROC_MAX_DROP:.4f} (att {att_cl:.4f}, float {fl_cl:.4f}) -> FAIL"
            failure_reasons.append(msg)

    checks["clean_auroc_loss_tolerance"] = {
        "passed": passed_cl,
        "pre_att": fl_cl,
        "post_att": att_cl,
        "drop": (fl_cl - att_cl) if fl_cl is not None and att_cl is not None else None,
        "max_allowed_drop": CLEAN_AUROC_MAX_DROP,
        "message": msg,
    }

    # 4. AI-positive recall >= 0.82
    att_ai_rec = att_metrics.get("ai_positive_recall")
    if att_ai_rec is None:
        passed_ai = False
        msg = f"[{track_name}] AI-positive recall missing from evaluation"
        failure_reasons.append(msg)
    elif att_ai_rec >= RECALL_MIN_THRESHOLD:
        passed_ai = True
        msg = f"[{track_name}] AI-positive recall: {att_ai_rec:.4f} >= {RECALL_MIN_THRESHOLD:.2f} -> PASS"
    else:
        passed_ai = False
        msg = f"[{track_name}] AI-positive recall: {att_ai_rec:.4f} < {RECALL_MIN_THRESHOLD:.2f} -> FAIL"
        failure_reasons.append(msg)

    checks["ai_positive_recall"] = {
        "passed": passed_ai,
        "value": att_ai_rec,
        "threshold": RECALL_MIN_THRESHOLD,
        "message": msg,
    }

    # 5. Authentic recall >= 0.82
    att_auth_rec = att_metrics.get("authentic_recall")
    if att_auth_rec is None:
        passed_auth = False
        msg = f"[{track_name}] Authentic recall missing from evaluation"
        failure_reasons.append(msg)
    elif att_auth_rec >= RECALL_MIN_THRESHOLD:
        passed_auth = True
        msg = f"[{track_name}] Authentic recall: {att_auth_rec:.4f} >= {RECALL_MIN_THRESHOLD:.2f} -> PASS"
    else:
        passed_auth = False
        msg = f"[{track_name}] Authentic recall: {att_auth_rec:.4f} < {RECALL_MIN_THRESHOLD:.2f} -> FAIL"
        failure_reasons.append(msg)

    checks["authentic_recall"] = {
        "passed": passed_auth,
        "value": att_auth_rec,
        "threshold": RECALL_MIN_THRESHOLD,
        "message": msg,
    }

    track_passed = all(c["passed"] for c in checks.values())

    return {
        "track": track_name,
        "passed": track_passed,
        "status": "PROMOTED" if track_passed else "REJECTED",
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def build_shared_report(tracks_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Construct shared multi-track promotion report."""
    all_passed = all(res["passed"] for res in tracks_results.values())
    return {
        "report_type": "att_promotion_gate_shared",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "passed": all_passed,
        "overall_status": "ALL_PROMOTED" if all_passed else "REJECTED",
        "tracks": tracks_results,
    }


def print_gate_summary(report: dict[str, Any]) -> None:
    """Print readable summary table of gate checks."""
    print("=" * 70)
    print(f"ATT PROMOTION GATE CHECK: {report.get('overall_status', 'UNKNOWN')}")
    print("=" * 70)
    for track_name, res in report.get("tracks", {}).items():
        status = res.get("status", "UNKNOWN")
        print(f"\n--- Track: {track_name.upper()} [{status}] ---")
        for check_name, check_data in res.get("checks", {}).items():
            symbol = "✓" if check_data.get("passed") else "✗"
            print(f"  [{symbol}] {check_data.get('message')}")
        if res.get("failure_reasons"):
            print("  Failures:")
            for reason in res["failure_reasons"]:
                print(f"    - {reason}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ATT promotion gate from evaluation JSONs")
    # Single track options
    parser.add_argument(
        "--float-eval", type=Path, default=None, help="Pre-ATT float student eval JSON"
    )
    parser.add_argument("--att-eval", type=Path, default=None, help="Post-ATT student eval JSON")
    parser.add_argument(
        "--track",
        type=str,
        default="student",
        choices=["small", "base", "student"],
        help="Track name",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output path for single track report"
    )

    # Shared dual-track options
    parser.add_argument(
        "--small-float-eval", type=Path, default=None, help="Small student pre-ATT float eval JSON"
    )
    parser.add_argument(
        "--small-att-eval", type=Path, default=None, help="Small student post-ATT eval JSON"
    )
    parser.add_argument(
        "--base-float-eval", type=Path, default=None, help="Base student pre-ATT float eval JSON"
    )
    parser.add_argument(
        "--base-att-eval", type=Path, default=None, help="Base student post-ATT eval JSON"
    )
    parser.add_argument(
        "--shared-report", type=Path, default=None, help="Output path for shared promotion report"
    )

    parser.add_argument(
        "--small-checkpoint", type=Path, default=None, help="Candidate checkpoint for small ATT"
    )
    parser.add_argument(
        "--base-checkpoint", type=Path, default=None, help="Candidate checkpoint for base ATT"
    )
    parser.add_argument(
        "--small-output-dir", type=Path, default=None, help="Output directory for small ATT"
    )
    parser.add_argument(
        "--base-output-dir", type=Path, default=None, help="Output directory for base ATT"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        default=False,
        help="Atomically promote candidate checkpoint if gate passes",
    )
    args = parser.parse_args()

    tracks_results: dict[str, dict[str, Any]] = {}

    # Dual-track shared evaluation mode
    if args.small_float_eval and args.small_att_eval:
        sm_fl = extract_gate_metrics(json.loads(args.small_float_eval.read_text(encoding="utf-8")))
        sm_att = extract_gate_metrics(json.loads(args.small_att_eval.read_text(encoding="utf-8")))
        tracks_results["small"] = evaluate_track_gate(sm_fl, sm_att, track_name="small")

    if args.base_float_eval and args.base_att_eval:
        ba_fl = extract_gate_metrics(json.loads(args.base_float_eval.read_text(encoding="utf-8")))
        ba_att = extract_gate_metrics(json.loads(args.base_att_eval.read_text(encoding="utf-8")))
        tracks_results["base"] = evaluate_track_gate(ba_fl, ba_att, track_name="base")

    # Single track fallback mode
    if not tracks_results and args.float_eval and args.att_eval:
        fl_metrics = extract_gate_metrics(json.loads(args.float_eval.read_text(encoding="utf-8")))
        att_metrics = extract_gate_metrics(json.loads(args.att_eval.read_text(encoding="utf-8")))
        tracks_results[args.track] = evaluate_track_gate(
            fl_metrics, att_metrics, track_name=args.track
        )

    if not tracks_results:
        print(
            "ERROR: Must provide either (--float-eval and --att-eval) or (--small-... and/or --base-...)",
            file=sys.stderr,
        )
        sys.exit(1)

    shared_report = build_shared_report(tracks_results)
    print_gate_summary(shared_report)

    # Save outputs
    if args.shared_report:
        args.shared_report.parent.mkdir(parents=True, exist_ok=True)
        args.shared_report.write_text(json.dumps(shared_report, indent=2), encoding="utf-8")
        print(f"Saved shared promotion report: {args.shared_report}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # If single track requested, save track result directly or wrapped
        single_track_name = (
            args.track if args.track in tracks_results else next(iter(tracks_results))
        )
        args.output.write_text(
            json.dumps(tracks_results[single_track_name], indent=2), encoding="utf-8"
        )
        print(f"Saved track report: {args.output}")

    if args.promote:
        track_configs = {
            "small": {
                "checkpoint": args.small_checkpoint,
                "output_dir": args.small_output_dir,
                "eval_json": args.small_att_eval,
            },
            "base": {
                "checkpoint": args.base_checkpoint,
                "output_dir": args.base_output_dir,
                "eval_json": args.base_att_eval,
            },
        }
        for track_name, track_result in tracks_results.items():
            track_config = track_configs[track_name]
            candidate = track_config["checkpoint"]
            output_dir = track_config["output_dir"]
            if candidate is None or output_dir is None:
                raise ValueError(
                    f"--promote requires --{track_name}-checkpoint and --{track_name}-output-dir"
                )
            candidate = Path(candidate)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            _, manifest_digest, threshold = load_evaluation_contract(track_config["eval_json"])
            promoted = output_dir / "checkpoint-promoted.pt"
            metadata = ATT_METADATA[track_name]

            if track_result["passed"]:
                atomic_copy_checkpoint(candidate, promoted)
                selected = promoted
                evaluation_status = "promoted"
                print(f"[{track_name.upper()}] Gate PASSED -> {promoted}")
            else:
                promoted.unlink(missing_ok=True)
                if not candidate.is_file():
                    raise FileNotFoundError(f"ATT candidate checkpoint not found: {candidate}")
                selected = candidate
                evaluation_status = "rejected"
                print(
                    f"[{track_name.upper()}] Gate FAILED. "
                    f"Reasons: {track_result['failure_reasons']}"
                )

            track_report = {
                "checkpoint_path": str(selected),
                "checkpoint_sha256": compute_file_sha256(selected),
                "manifest_digest": manifest_digest,
                "variant": track_name,
                "parameter_count": metadata["parameter_count"],
                "calibrated_threshold": threshold,
                "metrics": {"checks": track_result["checks"]},
                "passed": bool(track_result["passed"]),
                "failed_reasons": track_result["failure_reasons"],
            }
            (output_dir / "promotion_report.json").write_text(
                json.dumps(track_report, indent=2),
                encoding="utf-8",
            )
            metadata_sidecar = {
                "model_family": metadata["model_family"],
                "parameter_count": metadata["parameter_count"],
                "quantization": "float32",
                "calibrated_threshold": threshold,
                "input_size": [3, 224, 224],
                "preprocessing_version": 2,
                "manifest_digest": manifest_digest,
                "evaluation_status": evaluation_status,
            }
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata_sidecar, indent=2),
                encoding="utf-8",
            )
    sys.exit(0 if shared_report["passed"] else 1)


if __name__ == "__main__":
    main()

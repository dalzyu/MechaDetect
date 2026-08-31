#!/usr/bin/env python3
"""Check whether an evaluate_performance.py JSON output passes the promotion gate.

Gate thresholds (simplified for deadline):
  Clean AUROC             > 0.96
  AI-positive recall @0.5 > 0.82
  Authentic recall @0.5   > 0.82

Usage:
  python scripts/check_promotion_gate.py outputs/eval.json
  python scripts/check_promotion_gate.py outputs/eval.json --condition clean
  python scripts/check_promotion_gate.py outputs/eval.json --condition mean_robust
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AUROC_THRESHOLD = 0.96
RECALL_THRESHOLD = 0.82


def check_gate(
    metrics: dict[str, object], label: str = "clean"
) -> tuple[bool, list[str]]:
    passed = True
    messages = []

    auroc = metrics.get("ai_positive_auroc")
    ai_recall = metrics.get("binary_recall")
    auth_recall = metrics.get("authentic_recall")

    if auroc is None:
        messages.append(f"[{label}] AUROC is missing/undefined")
        passed = False
    elif auroc >= AUROC_THRESHOLD:
        messages.append(f"[{label}] AUROC:            {auroc:.4f}  >= {AUROC_THRESHOLD}  PASS")
    else:
        messages.append(f"[{label}] AUROC:            {auroc:.4f}  <  {AUROC_THRESHOLD}  FAIL")
        passed = False

    if ai_recall is None:
        messages.append(f"[{label}] AI-positive recall is missing/undefined")
        passed = False
    elif ai_recall >= RECALL_THRESHOLD:
        messages.append(f"[{label}] AI-positive recall: {ai_recall:.4f}  >= {RECALL_THRESHOLD}  PASS")
    else:
        messages.append(f"[{label}] AI-positive recall: {ai_recall:.4f}  <  {RECALL_THRESHOLD}  FAIL")
        passed = False

    if auth_recall is None:
        messages.append(f"[{label}] Authentic recall is missing/undefined")
        passed = False
    elif auth_recall >= RECALL_THRESHOLD:
        messages.append(f"[{label}] Authentic recall:   {auth_recall:.4f}  >= {RECALL_THRESHOLD}  PASS")
    else:
        messages.append(f"[{label}] Authentic recall:   {auth_recall:.4f}  <  {RECALL_THRESHOLD}  FAIL")
        passed = False

    return passed, messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Check promotion gate from evaluation JSON")
    parser.add_argument("eval_json", type=Path, help="Path to JSON output from evaluate_performance.py")
    parser.add_argument(
        "--condition",
        type=str,
        default="clean",
        help="Condition to check (default: clean, or any key in 'conditions')",
    )
    args = parser.parse_args()

    if not args.eval_json.is_file():
        print(f"ERROR: File not found: {args.eval_json}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.eval_json.read_text(encoding="utf-8"))
    conditions = data.get("conditions", {})

    if args.condition == "all":
        all_passed = True
        for cond_name, metrics in conditions.items():
            passed, msgs = check_gate(metrics, label=cond_name)
            for m in msgs:
                print(m)
            if not passed:
                all_passed = False
            print()
        status = "PROMOTED" if all_passed else "REJECTED"
        print(f"Overall status: {status}")
        sys.exit(0 if all_passed else 1)

    if args.condition not in conditions:
        print(
            f"ERROR: Condition '{args.condition}' not found. Available: {list(conditions.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    metrics = conditions[args.condition]
    passed, msgs = check_gate(metrics, label=args.condition)

    print("=" * 50)
    print(f"Promotion Gate Check — {args.eval_json.name} [{args.condition}]")
    print("=" * 50)
    for m in msgs:
        print(f"  {m}")
    print("=" * 50)
    print(f"Result: {'PROMOTED' if passed else 'REJECTED'}")
    print("=" * 50)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

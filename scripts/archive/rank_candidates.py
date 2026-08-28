from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric(value: object) -> float:
    return float(value) if value is not None else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluations", nargs="+", type=Path)
    args = parser.parse_args()
    candidates = []
    for path in args.evaluations:
        report = json.loads(path.read_text(encoding="utf-8"))
        clean = report["conditions"]["clean"]
        summary = report["selection_summary"]
        candidates.append(
            {
                "path": str(path),
                "heldout_aigc": metric(summary["mean_generator_auroc"]),
                "robustness": metric(summary["mean_transformed_auroc"]),
                "tamper": metric(clean["tamper_auroc"]),
                "throughput": metric(report["efficiency"]["images_per_second"]),
            }
        )
    fastest = max((candidate["throughput"] for candidate in candidates), default=1.0)
    for candidate in candidates:
        efficiency = candidate["throughput"] / fastest if fastest else 0.0
        candidate["score"] = (
            0.50 * candidate["heldout_aigc"]
            + 0.25 * candidate["robustness"]
            + 0.20 * candidate["tamper"]
            + 0.05 * efficiency
        )
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()

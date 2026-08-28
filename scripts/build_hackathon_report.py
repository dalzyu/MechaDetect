from __future__ import annotations

import argparse
import json
from pathlib import Path


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=int, default=610_919_189)
    args = parser.parse_args()
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    conditions = report["conditions"]
    efficiency = report["efficiency"]

    lines = [
        "# Robust AIGC Detection Evaluation",
        "",
        "## Clean and transformed performance",
        "",
        "| Condition | Fully-AIGC AUROC | AUROC drop | Binary accuracy | "
        "Binary balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in conditions.items():
        lines.append(
            f"| {name} | {percent(values['fully_aigc_auroc'])} | "
            f"{percent(values['fully_aigc_auroc_drop'])} | "
            f"{percent(values['binary_accuracy'])} | "
            f"{percent(values['binary_balanced_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Generator generalisation",
            "",
            "Each generator family is ranked against the same held-out non-AIGC pool.",
            "",
            "| Generator | AUROC | Accuracy |",
            "|---|---:|---:|",
        ]
    )
    for name, values in report["per_aigc_generator"].items():
        lines.append(
            f"| {name} | {percent(values['fully_aigc_auroc'])} | {percent(values['accuracy'])} |"
        )

    rows = report.get("rows", [])
    false_positives = sorted(
        (row for row in rows if row["target"] != "fully_aigc"),
        key=lambda row: row["probabilities"]["clean"][2],
        reverse=True,
    )[:5]
    false_negatives = sorted(
        (row for row in rows if row["target"] == "fully_aigc"),
        key=lambda row: row["probabilities"]["clean"][2],
    )[:5]
    lines.extend(
        [
            "",
            "## Error analysis",
            "",
            "### Highest-risk false positives",
            "",
            "| Image | Target | P(fully AIGC) |",
            "|---|---|---:|",
        ]
    )
    for row in false_positives:
        lines.append(
            f"| `{row['image_path']}` | {row['target']} | "
            f"{percent(row['probabilities']['clean'][2])} |"
        )
    lines.extend(
        [
            "",
            "### Highest-risk false negatives",
            "",
            "| Image | Generator | P(fully AIGC) |",
            "|---|---|---:|",
        ]
    )
    for row in false_negatives:
        lines.append(
            f"| `{row['image_path']}` | {row['generator']} | "
            f"{percent(row['probabilities']['clean'][2])} |"
        )

    lines.extend(
        [
            "",
            "## Feasibility",
            "",
            f"- Parameters: {args.parameters:,} "
            f"({args.parameters / 1e9:.3f}B; organizer limit: 2B).",
            f"- Inference: {efficiency['milliseconds_per_image']:.1f} ms/image "
            f"({efficiency['images_per_second']:.2f} images/s).",
            f"- Peak allocated VRAM: {efficiency['peak_allocated_vram_gib']:.2f} GiB.",
            f"- Clean inference views: {efficiency['views_on_clean']}.",
            "",
            "## Limitations and trade-offs",
            "",
            "- Detection is probabilistic and should not be treated as proof of provenance.",
            "- Severe blur and rescaling can erase generator evidence; "
            "condition-specific drops are shown above.",
            "- Three-view mode improves stability but costs roughly three times "
            "the Gemma inference work.",
            "- Thresholds must be calibrated to the deployment's false-positive tolerance.",
            "- The private organizer test set remains unseen and was not used for selection.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

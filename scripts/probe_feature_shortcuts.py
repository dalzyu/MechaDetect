from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def balanced_accuracy(target: torch.Tensor, prediction: torch.Tensor, classes: int) -> float:
    recalls = []
    for label in range(classes):
        selected = target == label
        if selected.any():
            recalls.append((prediction[selected] == label).float().mean())
    return float(torch.stack(recalls).mean())


def fit_probe(features: torch.Tensor, target: torch.Tensor, train: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    classes = int(target.max()) + 1
    mean = features[train].mean(0, keepdim=True)
    std = features[train].std(0, keepdim=True).clamp_min(1e-6)
    normalized = (features - mean) / std
    model = torch.nn.Linear(features.shape[1], classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-3)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(normalized[train]), target[train])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(normalized[test]).argmax(-1)
    return {
        "balanced_accuracy": balanced_accuracy(target[test], prediction, classes),
        "chance_balanced_accuracy": 1.0 / classes,
        "classes": classes,
        "test_rows": int(test.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear shortcut probes over frozen backbone tokens.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    vectors = []
    for index in range(len(frame)):
        payload = torch.load(args.cache / f"{index:05d}.pt", map_location="cpu", weights_only=False)
        vectors.append(payload["tokens"].float().mean(0))
    features = torch.stack(vectors)
    dataset_codes, datasets = pd.factorize(frame["dataset"], sort=True)
    ratio = frame["width"].astype(float) / frame["height"].astype(float)
    aspect = torch.tensor(((ratio > 1.1).astype(int) + (ratio < 0.9).astype(int) * 2).to_numpy())
    dataset_target = torch.tensor(dataset_codes)

    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(frame), generator=generator)
    split = round(len(frame) * 0.8)
    train = torch.zeros(len(frame), dtype=torch.bool)
    train[permutation[:split]] = True
    test = ~train
    result = {
        "dataset_labels": list(datasets),
        "dataset_probe": fit_probe(features, dataset_target, train, test),
        "aspect_probe": fit_probe(features, aspect, train, test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

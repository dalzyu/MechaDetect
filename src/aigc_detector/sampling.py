from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import Sampler

from .constants import Provenance

DEFAULT_CLASS_TARGETS = {
    int(Provenance.AUTHENTIC): 0.25,
    int(Provenance.TAMPERED): 0.25,
    int(Provenance.FULLY_AIGC): 0.50,
}


def generator_balanced_weights(
    provenance: Sequence[int],
    generators: Sequence[str],
    targets: Mapping[int, float] | None = None,
) -> torch.Tensor:
    if len(provenance) != len(generators):
        raise ValueError("provenance and generators must have equal lengths")
    targets = targets or DEFAULT_CLASS_TARGETS
    groups: dict[tuple[int, str], list[int]] = {}
    generators_per_class: dict[int, set[str]] = {}
    for index, (label, generator) in enumerate(zip(provenance, generators, strict=True)):
        groups.setdefault((int(label), generator), []).append(index)
        generators_per_class.setdefault(int(label), set()).add(generator)
    weights = torch.zeros(len(provenance), dtype=torch.double)
    for (label, _), indices in groups.items():
        if label not in targets:
            raise ValueError(f"No target probability for class {label}")
        group_mass = float(targets[label]) / len(generators_per_class[label])
        weights[indices] = group_mass / len(indices)
    return weights


def build_balanced_sampler(
    provenance: Sequence[int],
    generators: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> EpochWeightedSampler:
    return EpochWeightedSampler(
        generator_balanced_weights(provenance, generators), samples=samples, seed=seed
    )


class EpochWeightedSampler(Sampler[int]):
    """Replacement sampler whose complete order is a pure function of seed and epoch."""

    def __init__(self, weights: torch.Tensor, *, samples: int, seed: int) -> None:
        self.weights = weights
        self.samples = samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.samples, replacement=True, generator=generator
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.samples

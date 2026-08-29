from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import Sampler

from .constants import Provenance

DEFAULT_CLASS_TARGETS = {
    # Legacy three-way provenance targets retained for callers that have no
    # binary target column.
    int(Provenance.AUTHENTIC): 0.50,
    int(Provenance.TAMPERED): 0.25,
    int(Provenance.FULLY_AIGC): 0.25,
}


def generator_balanced_weights(
    provenance: Sequence[int],
    generators: Sequence[str],
    targets: Mapping[int, float] | None = None,
    *,
    ai_positive: Sequence[int] | None = None,
    max_ratio: float = 5.0,
) -> torch.Tensor:
    """Return deterministic capped hierarchical sampling weights.

    With ``ai_positive`` supplied, class mass is exactly 50/50 and generator
    groups receive inverse-frequency mass, capped at ``max_ratio`` relative to
    natural row sampling within each binary class.  The legacy provenance
    behavior remains available when no binary column is supplied.
    """
    if len(provenance) != len(generators):
        raise ValueError("provenance and generators must have equal lengths")
    if ai_positive is not None and len(ai_positive) != len(provenance):
        raise ValueError("ai_positive and provenance must have equal lengths")
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be at least 1")

    labels = [int(value) for value in (ai_positive if ai_positive is not None else provenance)]
    class_targets = {0: 0.50, 1: 0.50} if ai_positive is not None else (
        targets or DEFAULT_CLASS_TARGETS
    )
    groups: dict[tuple[int, str], list[int]] = {}
    class_counts: dict[int, int] = {}
    for index, (label, generator) in enumerate(zip(labels, generators, strict=True)):
        if label not in class_targets:
            raise ValueError(f"No target probability for class {label}")
        groups.setdefault((label, str(generator)), []).append(index)
        class_counts[label] = class_counts.get(label, 0) + 1
    if not class_counts:
        raise ValueError("Cannot build sampler for an empty manifest")

    # Equal group mass is useful for generator coverage, but tiny groups must
    # not dominate training.  Capping relative to natural class sampling
    # bounds each row's amplification while retaining exact class mass.
    weights = torch.zeros(len(labels), dtype=torch.double)
    group_total_by_class = {
        label: sum(1 for key in groups if key[0] == label)
        for label in class_targets
    }
    for (label, _generator), indices in groups.items():
        raw = float(class_targets[label]) / (
            group_total_by_class[label] * len(indices)
        )
        natural = float(class_targets[label]) / class_counts[label]
        capped = min(raw, max_ratio * natural)
        weights[indices] = capped
    for label, target in class_targets.items():
        class_indices = [index for index, value in enumerate(labels) if value == label]
        if not class_indices:
            continue
        mass = weights[class_indices].sum()
        if mass <= 0:
            raise ValueError(f"Class {label} has no positive sampling mass")
        weights[class_indices] *= float(target) / mass
    return weights


def build_balanced_sampler(
    provenance: Sequence[int],
    generators: Sequence[str],
    *,
    samples: int,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    ai_positive: Sequence[int] | None = None,
    max_ratio: float = 5.0,
) -> EpochWeightedSampler:
    return EpochWeightedSampler(
        generator_balanced_weights(
            provenance,
            generators,
            ai_positive=ai_positive,
            max_ratio=max_ratio,
        ),
        samples=samples,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )


class EpochWeightedSampler(Sampler[int]):
    """Replacement sampler whose complete order is a pure function of seed and epoch."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        samples: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside [0, {world_size})")
        self.weights = weights
        self.samples = samples
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.samples_per_rank = (samples + world_size - 1) // world_size
        self.epoch = 0
        self.start_offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_offset(self, offset: int) -> None:
        if not 0 <= offset <= self.samples_per_rank:
            raise ValueError(
                f"Sampler offset {offset} is outside [0, {self.samples_per_rank}]"
            )
        self.start_offset = offset

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        # Round up the global epoch length so every rank receives the same
        # number of samples. Equal rank lengths are required to avoid DDP
        # deadlocks when one GPU reaches backward() before another.
        total_size = self.samples_per_rank * self.world_size
        indices = torch.multinomial(
            self.weights, total_size, replacement=True, generator=generator
        )
        indices = indices[self.rank : total_size : self.world_size]
        return iter(indices[self.start_offset :].tolist())

    def __len__(self) -> int:
        return self.samples_per_rank - self.start_offset

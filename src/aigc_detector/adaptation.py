from __future__ import annotations

from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        base = self.base(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base + update * self.scaling


def apply_attention_lora(
    encoder: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> list[str]:
    targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    replaced = []
    for name, module in list(encoder.named_modules()):
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        linear = getattr(module, "linear", None)
        if not isinstance(linear, nn.Linear):
            continue
        module.linear = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        replaced.append(f"{name}.linear")
    if not replaced:
        raise RuntimeError("No Gemma attention projections accepted LoRA")
    return replaced


def trainable_encoder_state(encoder: nn.Module) -> dict[str, Tensor]:
    trainable = {name for name, parameter in encoder.named_parameters() if parameter.requires_grad}
    return {name: value for name, value in encoder.state_dict().items() if name in trainable}


def load_trainable_encoder_state(encoder: nn.Module, state: dict[str, Tensor]) -> None:
    current = encoder.state_dict()
    unexpected = sorted(set(state) - set(current))
    if unexpected:
        raise RuntimeError(f"Unexpected trainable encoder keys: {unexpected}")
    current.update(state)
    encoder.load_state_dict(current)

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


class ParameterEMA:
    """EMA over trainable parameters only; frozen Gemma weights are never duplicated."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, average in self.shadow.items():
            average.lerp_(parameters[name].detach(), 1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Any:
        parameters = dict(model.named_parameters())
        originals = {name: parameters[name].detach().clone() for name in self.shadow}
        training = model.training
        try:
            with torch.no_grad():
                for name, average in self.shadow.items():
                    parameters[name].copy_(average)
            model.eval()
            yield
        finally:
            with torch.no_grad():
                for name, original in originals.items():
                    parameters[name].copy_(original)
            model.train(training)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        shadow = state["shadow"]
        if not isinstance(shadow, dict):
            raise TypeError("EMA shadow state must be a dictionary")
        if set(shadow) != set(self.shadow):
            raise RuntimeError("EMA checkpoint parameters do not match the model")
        self.shadow = {name: value for name, value in shadow.items() if isinstance(value, Tensor)}


@torch.no_grad()
def load_ema_parameters(model: nn.Module, state: dict[str, object]) -> None:
    shadow = state.get("shadow")
    if not isinstance(shadow, dict):
        raise TypeError("EMA shadow state must be a dictionary")
    parameters = dict(model.named_parameters())
    missing = sorted(set(shadow) - set(parameters))
    if missing:
        raise RuntimeError(f"EMA checkpoint has unknown parameters: {missing}")
    for name, value in shadow.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"EMA value for {name} is not a tensor")
        parameters[name].copy_(value)

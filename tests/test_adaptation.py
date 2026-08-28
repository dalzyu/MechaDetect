import torch
from torch import nn

from aigc_detector.adaptation import LoRALinear, apply_attention_lora
from aigc_detector.train import build_optimizer


class Projection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = Projection()
        self.k_proj = Projection()
        self.v_proj = Projection()
        self.o_proj = Projection()
        self.unrelated = Projection()


def test_lora_only_touches_attention_projections() -> None:
    module = Attention()
    replaced = apply_attention_lora(module, rank=2, alpha=4, dropout=0.0)
    assert len(replaced) == 4
    assert isinstance(module.q_proj.linear, LoRALinear)
    assert isinstance(module.o_proj.linear, LoRALinear)
    assert isinstance(module.unrelated.linear, nn.Linear)
    output = module.q_proj.linear(torch.randn(3, 8)).sum()
    output.backward()
    assert module.q_proj.linear.lora_b.grad is not None
    assert module.q_proj.linear.base.weight.grad is None


class LayeredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.heads = nn.Linear(2, 1)


def test_optimizer_applies_layerwise_lr_decay() -> None:
    model = LayeredModel()
    optimizer = build_optimizer(
        model,
        {
            "training": {
                "heads_lr": 1.0e-4,
                "encoder_lr": 3.0e-6,
                "layerwise_lr_decay": 0.85,
                "weight_decay": 0.01,
            }
        },
    )
    assert [group["lr"] for group in optimizer.param_groups] == [
        1.0e-4,
        3.0e-6,
        3.0e-6 * 0.85,
        3.0e-6 * 0.85**2,
    ]

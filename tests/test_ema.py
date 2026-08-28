import torch
from torch import nn

from aigc_detector.ema import ParameterEMA, load_ema_parameters
from aigc_detector.losses import confidence_gated_kl


def test_ema_updates_and_temporarily_swaps_trainable_parameters() -> None:
    model = nn.Linear(2, 1)
    original = model.weight.detach().clone()
    ema = ParameterEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(2.0)
    changed = model.weight.detach().clone()
    ema.update(model)
    with ema.average_parameters(model):
        assert torch.allclose(model.weight, (original + changed) / 2)
    assert torch.allclose(model.weight, changed)


def test_confidence_gated_kl_ignores_uncertain_teacher() -> None:
    student = torch.tensor([[0.8, 0.1, 0.1]], requires_grad=True)
    teacher = torch.tensor([[0.4, 0.3, 0.3]])
    loss = confidence_gated_kl(student, teacher, threshold=0.8)
    assert loss.item() == 0.0
    loss.backward()
    assert student.grad is not None


def test_ema_checkpoint_can_be_loaded_as_inference_weights() -> None:
    source = nn.Linear(2, 1)
    ema = ParameterEMA(source)
    target = nn.Linear(2, 1)
    load_ema_parameters(target, ema.state_dict())
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)

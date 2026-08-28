import torch

from aigc_detector.constants import Provenance
from aigc_detector.losses import (
    hierarchical_classification_loss,
    mask_supervision_loss,
    provenance_robustness_loss,
)
from aigc_detector.model import ProvenanceHead, hierarchical_probabilities


def test_hierarchical_probabilities_sum_to_one() -> None:
    probabilities = hierarchical_probabilities(
        torch.tensor([-2.0, 0.0, 3.0]), torch.tensor([1.0, -1.0, 2.0])
    )
    assert probabilities.shape == (3, 3)
    assert torch.allclose(probabilities.sum(-1), torch.ones(3))
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_task_specific_heads_and_loss_backward() -> None:
    torch.manual_seed(42)
    heads = ProvenanceHead(encoder_dim=8, branch_dim=6, dropout=0.0)
    original = heads([torch.randn(7, 8), torch.randn(5, 8), torch.randn(6, 8)])
    transformed = heads([torch.randn(7, 8), torch.randn(5, 8), torch.randn(6, 8)])
    total, components = provenance_robustness_loss(
        original,
        transformed,
        provenance=torch.tensor([Provenance.AUTHENTIC, Provenance.TAMPERED, Provenance.FULLY_AIGC]),
    )
    assert torch.isfinite(total)
    assert "mask_dice" in components
    total.backward()
    assert heads.aigc_classifier.weight.grad is not None
    assert heads.tamper_classifier.weight.grad is not None


def test_fully_aigc_samples_are_excluded_from_tamper_loss() -> None:
    torch.manual_seed(1)
    heads = ProvenanceHead(encoder_dim=8, branch_dim=4, dropout=0.0)
    output = heads([torch.randn(5, 8)])
    loss = hierarchical_classification_loss(output, torch.tensor([Provenance.FULLY_AIGC]))
    tamper_gradient = torch.autograd.grad(loss, output.tamper_logit, allow_unused=True)[0]
    assert tamper_gradient is None or torch.count_nonzero(tamper_gradient) == 0


def test_fractional_mask_loss_accepts_soft_occupancy() -> None:
    logits = [torch.zeros(4, requires_grad=True)]
    focal, dice = mask_supervision_loss(logits, [torch.tensor([0.0, 0.25, 0.75, 1.0])])
    assert torch.isfinite(focal + dice)
    (focal + dice).backward()
    assert logits[0].grad is not None

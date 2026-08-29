import torch

from aigc_detector.constants import Provenance
from aigc_detector.losses import (
    ai_classification_loss,
    mask_supervision_loss,
    provenance_robustness_loss,
)
from aigc_detector.model import (
    ProvenanceHead,
    ai_generated_probability,
    binary_probabilities,
)


def test_binary_probabilities_sum_to_one() -> None:
    probabilities = binary_probabilities(torch.tensor([-2.0, 0.0, 3.0]))
    assert probabilities.shape == (3, 2)
    assert torch.allclose(probabilities.sum(-1), torch.ones(3))
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_track5_probability_is_binary_positive_probability() -> None:
    probabilities = torch.tensor(
        [
            [0.80, 0.20],
            [0.10, 0.90],
            [0.05, 0.95],
        ]
    )
    score = ai_generated_probability(probabilities)
    assert torch.allclose(score, torch.tensor([0.20, 0.90, 0.95]))


def test_binary_head_and_loss_backward() -> None:
    torch.manual_seed(42)
    heads = ProvenanceHead(encoder_dim=8, branch_dim=6, dropout=0.0)
    original = heads([torch.randn(7, 8), torch.randn(5, 8), torch.randn(6, 8)])
    transformed = heads([torch.randn(7, 8), torch.randn(5, 8), torch.randn(6, 8)])
    total, components = provenance_robustness_loss(
        original,
        transformed,
        provenance=torch.tensor(
            [Provenance.AUTHENTIC, Provenance.TAMPERED, Provenance.FULLY_AIGC]
        ),
    )
    assert torch.isfinite(total)
    assert "mask_dice" in components
    total.backward()
    assert heads.ai_positive_classifier.weight.grad is not None
    assert heads.token_tamper_classifier[1].weight.grad is not None


def test_ai_classification_treats_both_ai_subtypes_as_positive() -> None:
    torch.manual_seed(1)
    heads = ProvenanceHead(encoder_dim=8, branch_dim=4, dropout=0.0)
    output = heads([torch.randn(5, 8), torch.randn(5, 8)])
    labels = torch.tensor([Provenance.TAMPERED, Provenance.FULLY_AIGC])
    loss = ai_classification_loss(output, labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        output.ai_positive_logit,
        torch.ones(2),
    )
    assert torch.allclose(loss, expected)

def test_fractional_mask_loss_accepts_soft_occupancy() -> None:
    logits = [torch.zeros(4, requires_grad=True)]
    focal, dice = mask_supervision_loss(logits, [torch.tensor([0.0, 0.25, 0.75, 1.0])])
    assert torch.isfinite(focal + dice)
    (focal + dice).backward()
    assert logits[0].grad is not None

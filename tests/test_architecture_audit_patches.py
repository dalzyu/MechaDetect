import torch
import torch.nn.functional as F

from aigc_detector.constants import Provenance
from aigc_detector.losses import ai_classification_loss, confidence_gated_kl
from aigc_detector.model import ProvenanceHead, ProvenanceOutput, binary_probabilities
from scripts.distill_student import STUDENT_PRESETS, _slice_provenance_output, student_config


def test_human_tampered_images_train_as_negative_target() -> None:
    """Human memes with ai_positive=0 must be supervised as 0.0, even with provenance=TAMPERED."""
    output = ProvenanceOutput(
        ai_positive_logit=torch.tensor([2.0]),
        probabilities=torch.tensor([[0.12, 0.88]]),
        aigc_features=torch.zeros(1, 256),
        tamper_features=torch.zeros(1, 256),
        token_tamper_logits=[torch.zeros(10)],
    )
    # 1. When explicit ai_positive target is passed (0.0):
    loss_with_explicit_target = ai_classification_loss(output, torch.tensor([0.0]))
    expected_loss = F.binary_cross_entropy_with_logits(
        output.ai_positive_logit, torch.tensor([0.0])
    )
    assert torch.allclose(loss_with_explicit_target, expected_loss)
    # High loss because logit was positive (2.0) but target was 0.0
    assert loss_with_explicit_target.item() > 2.0
    # 2. When authentic label is passed:
    loss_authentic = ai_classification_loss(output, torch.tensor([Provenance.AUTHENTIC]))
    assert torch.allclose(loss_authentic, expected_loss)


def test_confidence_gated_kl_has_continuous_non_zero_gradients() -> None:
    """confidence_gated_kl must provide smooth gradients across extreme confident logits."""
    # Confident student logit (-15.0 -> prob ~ 3e-7)
    student_logit = torch.tensor([-15.0, 15.0], requires_grad=True)
    student_probs = binary_probabilities(student_logit)

    teacher_probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

    loss = confidence_gated_kl(
        student_probs,
        teacher_probs,
        threshold=0.8,
        student_logit=student_logit,
    )
    assert loss > 0.0
    loss.backward()
    # Gradient must be finite and non-zero (never truncated by clamp)
    assert student_logit.grad is not None
    assert torch.all(torch.isfinite(student_logit.grad))
    assert torch.all(student_logit.grad != 0.0)


def test_provenance_head_batched_matches_sequential() -> None:
    """Batched forward_batched_tokens must match sequential extract_features exactly."""
    torch.manual_seed(42)
    head = ProvenanceHead(encoder_dim=16, branch_dim=8, dropout=0.0).eval()

    # Batch of 2 images, each with 14 tokens
    tokens_tensor = torch.randn(2, 14, 16)
    tokens_list = [tokens_tensor[0], tokens_tensor[1]]

    with torch.no_grad():
        out_batched = head(tokens_tensor)
        out_seq = head(tokens_list)

    assert torch.allclose(out_batched.ai_positive_logit, out_seq.ai_positive_logit, atol=1e-5)
    assert torch.allclose(out_batched.probabilities, out_seq.probabilities, atol=1e-5)
    assert torch.allclose(out_batched.aigc_features, out_seq.aigc_features, atol=1e-5)
    assert torch.allclose(out_batched.tamper_features, out_seq.tamper_features, atol=1e-5)


def test_student_presets_and_loss_config_match_distillation_contract() -> None:
    teacher_config = {"model": {}, "training": {}, "loss": {}}
    expected = {
        "small": ("facebook/dinov3-vits16-pretrain-lvd1689m", 384),
        "base": ("facebook/dinov3-vitb16-pretrain-lvd1689m", 768),
    }
    for variant, (encoder_id, encoder_dim) in expected.items():
        config = student_config(teacher_config, variant)
        assert STUDENT_PRESETS[variant]["encoder_id"] == encoder_id
        assert STUDENT_PRESETS[variant]["encoder_dim"] == encoder_dim
        assert STUDENT_PRESETS[variant]["epochs"] == 2
        assert "max_updates" not in config["training"]
        assert config["model"]["backbone_type"] == "dinov3"
        assert config["model"]["encoder_id"] == encoder_id
        assert config["model"]["encoder_dim"] == encoder_dim
        assert config["training"]["heads_lr"] == 2.0e-4
        assert config["training"]["encoder_lr"] == 2.0e-5
        assert config["training"]["layerwise_lr_decay"] == 0.85
        assert config["training"]["weight_decay"] == 0.01
        assert config["training"]["required_world_size"] == 1
        assert config["training"]["epochs"] == 2
        assert (
            config["training"]["physical_batch_size"]
            * config["training"]["gradient_accumulation"]
            * config["training"]["required_world_size"]
            == 48
        )
        assert config["loss"] == {
            "provenance_original": 1.0,
            "provenance_transformed": 1.0,
            "prediction_consistency": 1.0,
            "feature_consistency": 0.5,
            "mask_focal": 0.0,
            "mask_dice": 0.0,
            "ema_consistency": 2.0,
            "teacher_feature_consistency": 0.5,
        }


def test_student_paired_output_slicing_preserves_all_provenance_fields() -> None:
    combined = ProvenanceOutput(
        ai_positive_logit=torch.arange(4, dtype=torch.float32),
        probabilities=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        aigc_features=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        tamper_features=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        token_tamper_logits=[torch.tensor([index], dtype=torch.float32) for index in range(4)],
        fusion_gates=torch.arange(8, dtype=torch.float32).reshape(4, 2),
    )
    original = _slice_provenance_output(combined, 0, 2)
    transformed = _slice_provenance_output(combined, 2)

    assert torch.equal(original.ai_positive_logit, combined.ai_positive_logit[:2])
    assert torch.equal(transformed.probabilities, combined.probabilities[2:])
    assert torch.equal(original.aigc_features, combined.aigc_features[:2])
    assert torch.equal(transformed.tamper_features, combined.tamper_features[2:])
    assert [value.item() for value in original.token_tamper_logits] == [0.0, 1.0]
    assert [value.item() for value in transformed.token_tamper_logits] == [2.0, 3.0]
    assert torch.equal(original.fusion_gates, combined.fusion_gates[:2])
    assert torch.equal(transformed.fusion_gates, combined.fusion_gates[2:])

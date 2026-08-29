from __future__ import annotations

"""Vision backbones, binary Track 5 heads, and optional edit localization.

Architecture overview:
    Input Image (PIL)
           │
           ▼
    Vision Backbone (DINOv3 ViT-H+, PE-Spatial-G, or Gemma 4)
           │  Produces a list of patch token sequences
           ▼
    Token Adapter (LayerNorm + Linear projection)
           │
           ├─────────────────────────────────────────────┐
           ▼                                             ▼
    Global AI-Evidence Head                        Edit Localization Head
    • Learned query vectors                         • Token-level classifier
    • Mean and standard-deviation pooling           • Top-5% patch pooling
    • MLP projection                                • Softmax-weighted pooling
           │                                             │
           └────────────────────┬────────────────────────┘
                                ▼
                 Binary Track 5 Classification Head
                 • P(AI-positive) = sigmoid(ai_positive_logit)
                 • P(authentic) = 1 - P(AI-positive)

The image-level objective is deliberately binary: authentic is negative, and
both fully generated and AI-edited images are positive. The localization head
can still use edit masks as an auxiliary task; it does not classify the two
positive subtypes against each other.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpointing
from PIL import Image
from torch import Tensor


def binary_probabilities(ai_positive_logit: Tensor) -> Tensor:
    """Compute [authentic, AI-positive] probabilities from one logit."""
    ai_positive = ai_positive_logit.sigmoid()
    return torch.stack((1.0 - ai_positive, ai_positive), dim=-1)


def ai_generated_probability(probabilities: Tensor) -> Tensor:
    """Return the Track 5 positive probability from binary probabilities."""
    if probabilities.shape[-1] != 2:
        raise ValueError("Expected [authentic, ai_positive] probabilities")
    return probabilities[..., 1]


@dataclass
class ProvenanceOutput:
    """Structured output for binary detection and optional localization.

    ``provenance`` remains the internal dataset label vocabulary, but the
    image-level prediction has only two classes: authentic and AI-positive.
    """

    ai_positive_logit: Tensor
    probabilities: Tensor
    aigc_features: Tensor
    tamper_features: Tensor
    token_tamper_logits: list[Tensor]
    fusion_gates: Tensor | None = None

    @property
    def provenance_logits(self) -> Tensor:
        """Return log probabilities for [authentic, ai_positive]."""
        return self.probabilities.clamp_min(1e-7).log()

    @property
    def provenance_features(self) -> Tensor:
        """Concatenate global and localization representations."""
        return torch.cat((self.aigc_features, self.tamper_features), dim=-1)

    @property
    def ai_generated_probability(self) -> Tensor:
        """Return P(AI-positive) for Track 5."""
        return ai_generated_probability(self.probabilities)

class LearnedQueryPool(nn.Module):
    """Pool variable-length token sequences into fixed queries using attention."""

    def __init__(self, encoder_dim: int, queries: int, heads: int = 4) -> None:
        super().__init__()
        if encoder_dim % heads != 0:
            raise ValueError(f"encoder_dim ({encoder_dim}) must be divisible by heads ({heads})")
        self.queries = nn.Parameter(torch.empty(queries, encoder_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(encoder_dim, heads, batch_first=True)

    def forward(self, tokens: Tensor) -> Tensor:
        """Pool a single image's token sequence [N, D] or batched [B, N, D]."""
        if tokens.dim() == 2:
            query = self.queries.to(dtype=tokens.dtype).unsqueeze(0)
            tokens_batch = tokens.unsqueeze(0)
            pooled, _ = self.attention(query, tokens_batch, tokens_batch, need_weights=False)
            return pooled.squeeze(0)
        B = tokens.shape[0]
        query = self.queries.to(dtype=tokens.dtype).unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return pooled


class ProvenanceHead(nn.Module):
    """Binary AI detector with an optional edit-localization branch.

    The global branch and token-aware branch produce complementary evidence.
    A single image-level classifier consumes both representations, so fully
    generated and AI-edited images share one positive target.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        trunk_dim: int = 512,
        branch_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Use trunk_dim if encoder_dim matches default, else use encoder_dim
        dim = encoder_dim
        self.encoder_dim = dim
        self.branch_dim = branch_dim

        # Global AI-evidence branch: learned queries + mean + standard deviation.
        self.aigc_queries = LearnedQueryPool(dim, queries=4)
        self.aigc_projection = nn.Sequential(
            nn.LayerNorm(dim * 6),
            nn.Linear(dim * 6, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Token-aware branch: localization evidence and optional mask supervision.
        self.token_tamper_classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )
        self.tamper_query = LearnedQueryPool(dim, queries=1)
        self.tamper_projection = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # One binary image-level decision; no fully-AIGC/tampered classifier.
        self.ai_positive_classifier = nn.Linear(branch_dim * 2, 1)


    def forward_batched_tokens(
        self, tokens: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Vectorized forward for ONNX export and WebGPU inference over [B, N, D]."""
        B, N, D = tokens.shape
        # Global AI-evidence branch
        query_tokens = self.aigc_queries(tokens).flatten(start_dim=1)
        mean_tokens = tokens.mean(dim=1)
        std_tokens = tokens.std(dim=1, unbiased=False)
        aigc_summary = torch.cat((query_tokens, mean_tokens, std_tokens), dim=-1)
        aigc_features = self.aigc_projection(aigc_summary)

        # Token-aware branch
        scores = self.token_tamper_classifier(tokens).squeeze(-1)
        attention_weights = scores.softmax(dim=-1).unsqueeze(-1)
        attention_avg = torch.sum(tokens * attention_weights, dim=1)
        global_query = self.tamper_query(tokens).squeeze(1)

        k = max(1, min(N, ceil(N * 0.05)))
        top_k_indices = scores.topk(k, dim=-1).indices
        top_k_tokens = torch.gather(tokens, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, D))
        top_k_avg = top_k_tokens.mean(dim=1)

        tamper_summary = torch.cat((top_k_avg, attention_avg, global_query), dim=-1)
        tamper_features = self.tamper_projection(tamper_summary)
        return aigc_features, tamper_features, scores

    def extract_features(
        self, token_sequences: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Extract global evidence, localization features, and patch scores."""
        if not token_sequences:
            raise ValueError("At least one image token sequence is required")

        aigc_features: list[Tensor] = []
        tamper_features: list[Tensor] = []
        patch_logits: list[Tensor] = []

        for raw_tokens in token_sequences:
            tokens = raw_tokens
            num_tokens = tokens.shape[0]

            query_tokens = self.aigc_queries(tokens).flatten()
            aigc_summary = torch.cat((query_tokens, tokens.mean(0), tokens.std(0, unbiased=False)))
            aigc_features.append(self.aigc_projection(aigc_summary))

            scores = self.token_tamper_classifier(tokens).squeeze(-1)
            keep_count = max(1, ceil(num_tokens * 0.05))
            top_k_indices = scores.topk(keep_count).indices
            top_k_avg = tokens[top_k_indices].mean(0)
            attention_avg = torch.sum(tokens * scores.softmax(0).unsqueeze(-1), dim=0)
            global_query = self.tamper_query(tokens).squeeze(0)

            tamper_summary = torch.cat((top_k_avg, attention_avg, global_query))
            tamper_features.append(self.tamper_projection(tamper_summary))
            patch_logits.append(scores)

        return torch.stack(aigc_features), torch.stack(tamper_features), patch_logits

    def forward(self, token_sequences: Sequence[Tensor] | Tensor) -> ProvenanceOutput:
        """Run binary classification across token sequences or batched tensor."""
        if isinstance(token_sequences, Tensor) and token_sequences.dim() == 3:
            aigc_features, tamper_features, scores = self.forward_batched_tokens(token_sequences)
            ai_positive_features = torch.cat((aigc_features, tamper_features), dim=-1)
            ai_positive_logit = self.ai_positive_classifier(ai_positive_features).squeeze(-1)
            return ProvenanceOutput(
                ai_positive_logit=ai_positive_logit,
                probabilities=binary_probabilities(ai_positive_logit),
                aigc_features=aigc_features,
                tamper_features=tamper_features,
                token_tamper_logits=[scores[i] for i in range(scores.shape[0])],
            )

        aigc_features, tamper_features, patch_logits = self.extract_features(token_sequences)
        ai_positive_features = torch.cat((aigc_features, tamper_features), dim=-1)
        ai_positive_logit = self.ai_positive_classifier(ai_positive_features).squeeze(-1)
        return ProvenanceOutput(
            ai_positive_logit=ai_positive_logit,
            probabilities=binary_probabilities(ai_positive_logit),
            aigc_features=aigc_features,
            tamper_features=tamper_features,
            token_tamper_logits=patch_logits,
        )


# Backward-compatible alias
TaskSpecificProvenanceHead = ProvenanceHead


class SpectralExpert(nn.Module):
    """Frequency-domain expert using ConvNeXt over spatial residuals + radial 2D FFT."""

    def __init__(
        self,
        output_dim: int = 256,
        image_size: int = 384,
        fft_bins: int = 32,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        network = convnext_tiny(weights=weights)

        # Adapt ConvNeXt 3-channel stem to 6 channels (RGB + 3 spatial residual filters)
        original_stem = network.features[0][0]
        replacement = nn.Conv2d(
            6,
            original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
        )
        with torch.no_grad():
            replacement.weight[:, :3].copy_(original_stem.weight)
            replacement.weight[:, 3:].copy_(original_stem.weight.mean(dim=1, keepdim=True))
            if original_stem.bias is not None:
                replacement.bias.copy_(original_stem.bias)
        network.features[0][0] = replacement
        network.classifier[-1] = nn.Linear(network.classifier[-1].in_features, output_dim)
        self.network = network
        self.image_size = image_size
        self.fft_bins = fft_bins

        # Radial FFT projection: 32 frequency energy bins -> 128 -> output_dim
        self.fft_projection = nn.Sequential(
            nn.Linear(fft_bins, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

        # Fixed high-pass spatial filtering kernels: horizontal, vertical, and Laplacian
        kernels = torch.tensor(
            [
                [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("residual_kernels", kernels, persistent=True)

    def _images_to_tensor(self, images: Sequence[Image.Image], device: torch.device) -> Tensor:
        arrays = [
            np.asarray(img.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC))
            for img in images
        ]
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        return tensor.div_(255.0)

    def _radial_fft(self, rgb: Tensor) -> Tensor:
        gray = rgb.mean(dim=1)
        spectrum = torch.log1p(torch.fft.rfft2(gray, norm="ortho").abs())
        spectrum = torch.fft.fftshift(spectrum, dim=(-2,))
        h, w = spectrum.shape[-2:]
        y = torch.linspace(-1.0, 1.0, h, device=rgb.device)
        x = torch.linspace(0.0, 1.0, w, device=rgb.device)
        radius = torch.sqrt(y[:, None].square() + x[None, :].square()).clamp_max(1.0)
        bin_indices = torch.clamp((radius * self.fft_bins).long(), max=self.fft_bins - 1)

        vectors = []
        for i in range(self.fft_bins):
            mask = bin_indices == i
            if mask.any():
                vectors.append(spectrum[:, mask].mean(dim=1))
            else:
                vectors.append(torch.zeros(spectrum.shape[0], device=spectrum.device))
        return torch.stack(vectors, dim=-1)

    def forward(self, images: Sequence[Image.Image]) -> Tensor:
        device = next(self.parameters()).device
        rgb = self._images_to_tensor(images, device)
        gray = rgb.mean(dim=1, keepdim=True)
        residuals = F.conv2d(gray, self.residual_kernels, padding=1)
        residuals = residuals.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        spatial_features = self.network(torch.cat((rgb, residuals), dim=1))
        frequency_features = self.fft_projection(self._radial_fft(rgb))
        return spatial_features + frequency_features


class DINOv3VisionBackbone(nn.Module):
    """DINOv3 ViT-H+/16 backbone wrapper (Tournament Winner).

    Input: $224 \times 224$ images processed via AutoImageProcessor.
    Output: $14 \times 14 = 196$ spatial patch tokens of dimension 1280.
    Prefix tokens (1 CLS + 4 register tokens = 5) are sliced off.
    """

    def __init__(
        self,
        encoder_id: str,
        *,
        revision: str | None,
        image_size: int = 224,
        freeze: bool = True,
        dtype: torch.dtype = torch.float32,  # fp32 master weights; autocast handles bf16 compute
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoImageProcessor, DINOv3ViTModel
        except ImportError as exc:
            raise RuntimeError("DINOv3 requires transformers>=5.10.1") from exc

        self.processor = AutoImageProcessor.from_pretrained(
            encoder_id, revision=revision, size={"height": image_size, "width": image_size}
        )
        self.encoder = DINOv3ViTModel.from_pretrained(encoder_id, revision=revision, dtype=dtype)
        self.set_frozen(freeze)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def set_frozen(self, frozen: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not frozen

    def set_trainable_last_layers(self, count: int) -> None:
        """Unfreeze the last `count` transformer layers (e.g. 8 of 32 for last-quarter adaptation)."""
        layers = self.encoder.model.layer
        if not 0 <= count <= len(layers):
            raise ValueError(f"Requested {count} trainable layers; encoder has {len(layers)}")
        self.set_frozen(True)
        for layer in layers[len(layers) - count :] if count else []:
            for param in layer.parameters():
                param.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing_enable()

    def forward(self, images: Sequence[Image.Image]) -> list[Tensor]:
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        is_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        with torch.no_grad() if is_frozen else nullcontext():
            # Slice off the first 5 tokens (1 CLS + 4 register tokens)
            patch_tokens = self.encoder(pixel_values=pixels).last_hidden_state[:, 5:]
        return list(patch_tokens)


class PESpatialVisionBackbone(nn.Module):
    """PE-Spatial-G/14 backbone wrapper (Tournament Runner-Up).

    Input: $448 \times 448$ images squashed with bilinear interpolation and normalized [-1, 1].
    Output: $32 \times 32 = 1024$ spatial patch tokens of dimension 1536 (no CLS token).
    """

    def __init__(
        self,
        encoder_id: str,
        *,
        revision: str | None,
        image_size: int = 448,
        freeze: bool = True,
        dtype: torch.dtype = torch.float32,  # fp32 master weights; autocast handles bf16 compute
    ) -> None:
        super().__init__()
        try:
            from core.vision_encoder.pe import VisionTransformer
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("PE-Spatial requires perception_models and huggingface-hub") from exc

        weights_path = hf_hub_download(encoder_id, filename="PE-Spatial-G14-448.pt", revision=revision)
        self.encoder = VisionTransformer.from_config(
            "PE-Spatial-G14-448",
            pretrained=True,
            checkpoint_path=weights_path,
            image_size=image_size,
        )
        self.encoder.to(dtype=dtype)
        self.image_size = image_size
        self.set_frozen(freeze)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def set_frozen(self, frozen: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not frozen

    def set_trainable_last_layers(self, count: int) -> None:
        """Unfreeze the final `count` residual attention blocks (e.g. 13 of 50 for last quarter)."""
        blocks = self.encoder.transformer.resblocks
        if not 0 <= count <= len(blocks):
            raise ValueError(f"Requested {count} trainable layers; encoder has {len(blocks)}")
        self.set_frozen(True)
        for block in blocks[len(blocks) - count :] if count else []:
            for param in block.parameters():
                param.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        transformer = self.encoder.transformer
        transformer.grad_checkpointing = True

        def checkpointed_forward(hidden: Tensor, attn_mask: Tensor | None = None, layer_idx: int = -1) -> Tensor:
            stop_idx = (transformer.layers + layer_idx) % transformer.layers
            for i, block in enumerate(transformer.resblocks):
                hidden = checkpointing.checkpoint(block, hidden, attn_mask, use_reentrant=False)
                if i == stop_idx:
                    break
            return hidden

        transformer.forward = checkpointed_forward

    def forward(self, images: Sequence[Image.Image]) -> list[Tensor]:
        arrays = [
            np.asarray(img.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR))
            for img in images
        ]
        tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(
            device=self.device, dtype=next(self.encoder.parameters()).dtype
        )
        # Normalize to [-1.0, 1.0] as trained in perception_models
        pixels = tensor.div_(255.0).sub_(0.5).div_(0.5)
        is_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        with torch.no_grad() if is_frozen else nullcontext():
            tokens = self.encoder.forward_features(pixels, norm=False)
        return list(tokens)


class Gemma4VisionBackbone(nn.Module):
    """Gemma 4 vision tower wrapper (Tournament Third Place)."""

    def __init__(
        self,
        encoder_id: str,
        *,
        revision: str | None,
        visual_tokens: int = 1120,
        freeze: bool = True,
        dtype: torch.dtype = torch.float32,  # fp32 master weights; autocast handles bf16 compute
    ) -> None:
        super().__init__()
        try:
            from transformers import Gemma4ImageProcessor, Gemma4VisionModel
        except ImportError as exc:
            raise RuntimeError("Gemma 4 requires transformers==5.10.1") from exc

        self.visual_tokens = visual_tokens
        self.processor = Gemma4ImageProcessor.from_pretrained(encoder_id, revision=revision)
        self.encoder = Gemma4VisionModel.from_pretrained(encoder_id, revision=revision, dtype=dtype)
        self.set_frozen(freeze)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def set_frozen(self, frozen: bool) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = not frozen

    def set_trainable_last_layers(self, count: int) -> None:
        layers = self.encoder.encoder.layers
        if not 0 <= count <= len(layers):
            raise ValueError(f"Requested {count} trainable layers; encoder has {len(layers)}")
        self.set_frozen(True)
        for layer in layers[len(layers) - count :] if count else []:
            for param in layer.parameters():
                param.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def forward(self, images: Sequence[Image.Image]) -> list[Tensor]:
        processed = self.processor(
            images=list(images), max_soft_tokens=self.visual_tokens, return_tensors="pt"
        )
        counts = [int(v) for v in processed["num_soft_tokens_per_image"].reshape(-1)]
        pixels = processed["pixel_values"].to(self.device)
        positions = processed["image_position_ids"].to(self.device)
        is_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        with torch.no_grad() if is_frozen else nullcontext():
            output = self.encoder(pixel_values=pixels, pixel_position_ids=positions)
        hidden = output.last_hidden_state
        return list(hidden.split(counts, dim=0))


class HFViTVisionBackbone(nn.Module):
    """Generic Hugging Face ViT/DINOv2 wrapper for student encoders."""

    def __init__(
        self,
        encoder_id: str,
        *,
        revision: str | None,
        image_size: int,
        freeze: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("The generic ViT backbone requires transformers") from exc
        self.processor = AutoImageProcessor.from_pretrained(
            encoder_id, revision=revision, size={"height": image_size, "width": image_size}
        )
        self.encoder = AutoModel.from_pretrained(encoder_id, revision=revision, dtype=dtype)
        self.set_frozen(freeze)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def set_frozen(self, frozen: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen

    def set_trainable_last_layers(self, count: int) -> None:
        layers = getattr(self.encoder, "encoder", self.encoder).layer
        if not 0 <= count <= len(layers):
            raise ValueError(f"Requested {count} trainable layers; encoder has {len(layers)}")
        self.set_frozen(True)
        for layer in layers[len(layers) - count :] if count else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing_enable()

    def forward(self, images: Sequence[Image.Image]) -> list[Tensor]:
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(self.device)
        frozen = not any(parameter.requires_grad for parameter in self.encoder.parameters())
        with torch.no_grad() if frozen else nullcontext():
            hidden = self.encoder(pixel_values=pixels).last_hidden_state
        # ViT and DINOv2 expose one CLS token; retain only spatial tokens.
        return list(hidden[:, 1:])

def build_backbone(
    backbone_type: str,
    encoder_id: str,
    *,
    encoder_revision: str | None,
    visual_tokens: int = 1120,
    image_size: int = 224,
    freeze: bool = True,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Factory function for instantiating the selected vision backbone."""
    if backbone_type == "dinov3":
        return DINOv3VisionBackbone(
            encoder_id,
            revision=encoder_revision,
            image_size=image_size,
            freeze=freeze,
            dtype=dtype,
        )
    if backbone_type == "vit":
        return HFViTVisionBackbone(
            encoder_id,
            revision=encoder_revision,
            image_size=image_size,
            freeze=freeze,
            dtype=dtype,
        )
    if backbone_type == "pe_spatial":
        return PESpatialVisionBackbone(
            encoder_id,
            revision=encoder_revision,
            image_size=image_size,
            freeze=freeze,
            dtype=dtype,
        )
    if backbone_type == "gemma4":
        return Gemma4VisionBackbone(
            encoder_id,
            revision=encoder_revision,
            visual_tokens=visual_tokens,
            freeze=freeze,
            dtype=dtype,
        )
    raise ValueError(
        f"Unsupported backbone_type {backbone_type!r}; use dinov3, vit, pe_spatial, or gemma4."
    )


class ProvenanceModel(nn.Module):
    """Complete binary Track 5 detector with optional spectral fusion."""

    def __init__(
        self,
        encoder_id: str,
        *,
        encoder_revision: str | None = None,
        backbone_type: str = "dinov3",
        visual_tokens: int = 1120,
        image_size: int = 224,
        encoder_dim: int = 1280,
        trunk_dim: int = 512,
        branch_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        spectral_expert: bool = False,
        spectral_image_size: int = 384,
        spectral_pretrained: bool = False,
        use_token_adapter: bool = True,
        encoder_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            backbone_type,
            encoder_id,
            encoder_revision=encoder_revision,
            visual_tokens=visual_tokens,
            image_size=image_size,
            freeze=freeze_encoder,
            dtype=encoder_dtype,
        )
        # Common 512-dim linear projection adapter (ensures architectural fairness across backbones)
        self.token_adapter = (
            nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, trunk_dim),
            )
            if use_token_adapter
            else nn.Identity()
        )
        head_input_dim = trunk_dim if use_token_adapter else encoder_dim
        self.heads = ProvenanceHead(
            encoder_dim=head_input_dim,
            trunk_dim=trunk_dim,
            branch_dim=branch_dim,
            dropout=dropout,
        )
        self.spectral = (
            SpectralExpert(
                output_dim=branch_dim,
                image_size=spectral_image_size,
                pretrained=spectral_pretrained,
            )
            if spectral_expert
            else None
        )
        if self.spectral is not None:
            self.aigc_gate = nn.Linear(branch_dim * 2, 1)
            self.tamper_gate = nn.Linear(branch_dim * 2, 1)

    def forward_tokens(self, token_sequences: Sequence[Tensor]) -> ProvenanceOutput:
        """Execute adapter and provenance heads directly on pre-extracted backbone tokens.

        Bypasses backbone computation when features are cached on disk during frozen screening.
        """
        adapted = [self.token_adapter(tokens.float()) for tokens in token_sequences]
        return self.heads(adapted)

    def forward(self, images: Sequence[Image.Image]) -> ProvenanceOutput:
        """Run binary inference from PIL images."""
        backbone_tokens = self.backbone(images)
        output = self.forward_tokens(backbone_tokens)
        if self.spectral is None:
            return output

        # Dynamic frequency-domain fusion into both complementary evidence branches.
        spectral = self.spectral(images)
        aigc_gate = torch.sigmoid(self.aigc_gate(torch.cat((output.aigc_features, spectral), -1)))
        tamper_gate = torch.sigmoid(self.tamper_gate(torch.cat((output.tamper_features, spectral), -1)))
        fused_aigc = aigc_gate * output.aigc_features + (1.0 - aigc_gate) * spectral
        fused_tamper = tamper_gate * output.tamper_features + (1.0 - tamper_gate) * spectral

        ai_positive_features = torch.cat((fused_aigc, fused_tamper), dim=-1)
        ai_positive_logit = self.heads.ai_positive_classifier(ai_positive_features).squeeze(-1)
        return ProvenanceOutput(
            ai_positive_logit=ai_positive_logit,
            probabilities=binary_probabilities(ai_positive_logit),
            aigc_features=fused_aigc,
            tamper_features=fused_tamper,
            token_tamper_logits=output.token_tamper_logits,
            fusion_gates=torch.cat((aigc_gate, tamper_gate), dim=-1),
        )
    def forward_tensor(self, pixel_values: Tensor) -> Tensor:
        """Run batched tensor inference [B, 3, H, W] returning binary ai_positive_logit [B] for ONNX/WebGPU."""
        if hasattr(self.backbone, "encoder"):
            raw_hidden = self.backbone.encoder(pixel_values=pixel_values).last_hidden_state
            tokens = raw_hidden[:, 5:] if raw_hidden.shape[1] > 5 else raw_hidden
        else:
            raise NotImplementedError("Direct tensor forward requires encoder backbone")

        adapted = self.token_adapter(tokens)
        aigc_features, tamper_features, _ = self.heads.forward_batched_tokens(adapted)
        ai_positive_features = torch.cat((aigc_features, tamper_features), dim=-1)
        return self.heads.ai_positive_classifier(ai_positive_features).squeeze(-1)

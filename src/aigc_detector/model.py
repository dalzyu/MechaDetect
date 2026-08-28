from __future__ import annotations

"""Vision backbones, common token adapters, and dual-task provenance heads.

Architecture overview:
    Input Image (PIL)
           │
           ▼
    Vision Backbone (DINOv3 ViT-H+, PE-Spatial-G, or Gemma 4)
           │  Produces a list of patch token sequences: list[Tensor of shape [N, D_enc]]
           ▼
    Token Adapter (LayerNorm + Linear projection: D_enc -> D_trunk, default 512)
           │  Produces normalized token sequences: list[Tensor of shape [N, 512]]
           ├──────────────────────────────────────────┐
           ▼                                          ▼
    AIGC Learned Query Head                    Token-Aware Tamper Head
    • 4 learned query vectors                  • Token-level linear classifier
    • Multi-head cross-attention               • Top-5% patch pooling
    • Mean & standard deviation pooling        • Softmax-weighted attention pooling
    • MLP projection -> aigc_logit             • MLP projection -> tamper_logit
           │                                          │
           └────────────────────┬─────────────────────┘
                                ▼
               Hierarchical Probabilities Engine
               • P(fully_aigc) = sigmoid(aigc_logit)
               • P(tampered)   = (1 - P(fully_aigc)) * sigmoid(tamper_logit)
               • P(authentic)  = (1 - P(fully_aigc)) * (1 - sigmoid(tamper_logit))

Why hierarchical probabilities?
    In standard 3-class softmax, P(tampered) directly competes with P(fully_aigc).
    When generative models produce unnatural high-frequency artifacts (such as warped
    fingers or inconsistent backgrounds), flat 3-class classifiers often mistake the
    synthetic image for a locally tampered authentic photograph.
    
    The hierarchical formulation decouples these two questions:
    1. Global Decision: Was the image synthesized end-to-end by a generative model?
    2. Localized Decision: If the image is a real photograph, was any region edited?
    
    Mathematically, P(authentic) + P(tampered) + P(fully_aigc) == 1.0 at all times.
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


def hierarchical_probabilities(aigc_logit: Tensor, tamper_logit: Tensor) -> Tensor:
    """Compute mutually exclusive [authentic, tampered, fully_aigc] probabilities.

    Args:
        aigc_logit: Unconstrained logit for global AI generation [B] or [B, 1].
        tamper_logit: Unconstrained logit for localized manipulation [B] or [B, 1].

    Returns:
        Tensor of shape [B, 3] where columns represent:
            index 0: P(authentic)  = (1 - a) * (1 - t)
            index 1: P(tampered)   = (1 - a) * t
            index 2: P(fully_aigc) = a
        where a = sigmoid(aigc_logit) and t = sigmoid(tamper_logit).
        Probabilities sum to 1.0 along the last dimension.
    """
    aigc = aigc_logit.sigmoid()
    tamper = tamper_logit.sigmoid()
    return torch.stack(((1.0 - aigc) * (1.0 - tamper), (1.0 - aigc) * tamper, aigc), dim=-1)


@dataclass
class ProvenanceOutput:
    """Structured container for all outputs produced during model inference.

    Attributes:
        aigc_logit: Raw scalar logit for global synthetic generation [B].
        tamper_logit: Raw scalar logit for localized tampering [B].
        probabilities: Normalized 3-class probabilities [B, 3] (authentic, tampered, aigc).
        aigc_features: Dense feature vector from the AIGC expert [B, branch_dim].
        tamper_features: Dense feature vector from the tamper expert [B, branch_dim].
        token_tamper_logits: Per-token tampering predictions for each sample in the batch.
                             List of length B, with each tensor shaped [N_tokens].
        fusion_gates: Gating values when the optional spectral expert is active [B, 2].
    """

    aigc_logit: Tensor
    tamper_logit: Tensor
    probabilities: Tensor
    aigc_features: Tensor
    tamper_features: Tensor
    token_tamper_logits: list[Tensor]
    fusion_gates: Tensor | None = None

    @property
    def provenance_logits(self) -> Tensor:
        """Compatibility view for evaluators; softmax(provenance_logits) matches probabilities."""
        return self.probabilities.clamp_min(1e-7).log()

    @property
    def provenance_features(self) -> Tensor:
        """Concatenated representation across both tasks [B, branch_dim * 2]."""
        return torch.cat((self.aigc_features, self.tamper_features), dim=-1)


class LearnedQueryPool(nn.Module):
    """Pools variable-length token sequences into fixed queries using multi-head attention.

    Learned query parameters cross-attend over the spatial patch tokens, allowing the
    model to extract fixed-size global summaries invariant to image resolution or token count.
    """

    def __init__(self, encoder_dim: int, queries: int, heads: int = 4) -> None:
        super().__init__()
        if encoder_dim % heads != 0:
            raise ValueError(f"encoder_dim ({encoder_dim}) must be divisible by heads ({heads})")
        self.queries = nn.Parameter(torch.empty(queries, encoder_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(encoder_dim, heads, batch_first=True)

    def forward(self, tokens: Tensor) -> Tensor:
        """Pool a single image's token sequence.

        Args:
            tokens: Patch embeddings of shape [num_tokens, encoder_dim].

        Returns:
            Pooled query embeddings of shape [queries, encoder_dim].
        """
        query = self.queries.to(dtype=tokens.dtype).unsqueeze(0)
        tokens_batch = tokens.unsqueeze(0)
        pooled, _ = self.attention(query, tokens_batch, tokens_batch, need_weights=False)
        return pooled.squeeze(0)


class ProvenanceHead(nn.Module):
    """Dual-task provenance classification head over visual patch tokens.

    Branch 1 (Global AIGC Detection):
        Combines 4 learned attention queries with global mean and standard deviation
        pooling across all visual tokens (effective dimension = encoder_dim * 6).
        Projects to branch_dim (256) through LayerNorm + GELU + Dropout.

    Branch 2 (Localized Tamper Detection):
        Scores every individual patch token using a linear classifier. Computes:
        1. Top-5% patch average (strongest suspicious regions).
        2. Softmax-weighted attention average across all patches.
        3. One learned global query.
        Concatenates these 3 vectors (effective dimension = encoder_dim * 3) and
        projects to branch_dim (256).
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
        dim = trunk_dim if encoder_dim == 512 else encoder_dim
        self.encoder_dim = dim
        self.branch_dim = branch_dim

        # Branch 1: Global AI Generation Expert
        self.aigc_queries = LearnedQueryPool(dim, queries=4)
        self.aigc_projection = nn.Sequential(
            nn.LayerNorm(dim * 6),
            nn.Linear(dim * 6, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aigc_classifier = nn.Linear(branch_dim, 1)

        # Branch 2: Localized Tamper Expert
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
        self.tamper_classifier = nn.Linear(branch_dim, 1)

    def extract_features(
        self, token_sequences: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Extract task-specific feature vectors and patch scores from tokens.

        Args:
            token_sequences: List of length B, where each element is [N_tokens, dim].

        Returns:
            Tuple of:
                - aigc_features: [B, branch_dim]
                - tamper_features: [B, branch_dim]
                - patch_logits: list of B tensors, each [N_tokens]
        """
        if not token_sequences:
            raise ValueError("At least one image token sequence is required")

        aigc_features: list[Tensor] = []
        tamper_features: list[Tensor] = []
        patch_logits: list[Tensor] = []

        for raw_tokens in token_sequences:
            tokens = raw_tokens.float()
            num_tokens = tokens.shape[0]

            # 1. AIGC Feature Extraction: 4 queries (4*D) + mean (1*D) + std (1*D) = 6*D
            query_tokens = self.aigc_queries(tokens).flatten()
            aigc_summary = torch.cat((query_tokens, tokens.mean(0), tokens.std(0, unbiased=False)))
            aigc_features.append(self.aigc_projection(aigc_summary))

            # 2. Tamper Feature Extraction: Top-k suspicious patches + attention avg + query
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

    def forward(self, token_sequences: Sequence[Tensor]) -> ProvenanceOutput:
        """Run classification across the batch of token sequences."""
        aigc_features, tamper_features, patch_logits = self.extract_features(token_sequences)
        aigc_logit = self.aigc_classifier(aigc_features).squeeze(-1)
        tamper_logit = self.tamper_classifier(tamper_features).squeeze(-1)
        return ProvenanceOutput(
            aigc_logit=aigc_logit,
            tamper_logit=tamper_logit,
            probabilities=hierarchical_probabilities(aigc_logit, tamper_logit),
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
            vectors.append(spectrum[:, mask].mean(dim=1))
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
        dtype: torch.dtype = torch.bfloat16,
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
        dtype: torch.dtype = torch.bfloat16,
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
        dtype: torch.dtype = torch.bfloat16,
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


def build_backbone(
    backbone_type: str,
    encoder_id: str,
    *,
    encoder_revision: str | None,
    visual_tokens: int = 1120,
    image_size: int = 224,
    freeze: bool = True,
) -> nn.Module:
    """Factory function for instantiating the selected vision backbone."""
    if backbone_type == "dinov3":
        return DINOv3VisionBackbone(
            encoder_id,
            revision=encoder_revision,
            image_size=image_size,
            freeze=freeze,
        )
    if backbone_type == "pe_spatial":
        return PESpatialVisionBackbone(
            encoder_id,
            revision=encoder_revision,
            image_size=image_size,
            freeze=freeze,
        )
    if backbone_type == "gemma4":
        return Gemma4VisionBackbone(
            encoder_id,
            revision=encoder_revision,
            visual_tokens=visual_tokens,
            freeze=freeze,
        )
    raise ValueError(f"Unsupported backbone_type: {backbone_type!r}. Must be 'dinov3', 'pe_spatial', or 'gemma4'.")


class ProvenanceModel(nn.Module):
    """Complete provenance detection model: Backbone -> Adapter -> Dual Heads (+ Spectral Expert)."""

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
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            backbone_type,
            encoder_id,
            encoder_revision=encoder_revision,
            visual_tokens=visual_tokens,
            image_size=image_size,
            freeze=freeze_encoder,
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
        """Full end-to-end forward pass from PIL Images to hierarchical probabilities."""
        backbone_tokens = self.backbone(images)
        output = self.forward_tokens(backbone_tokens)
        if self.spectral is None:
            return output

        # Dynamic frequency-domain fusion via learned sigmoid gates
        spectral = self.spectral(images)
        aigc_gate = torch.sigmoid(self.aigc_gate(torch.cat((output.aigc_features, spectral), -1)))
        tamper_gate = torch.sigmoid(self.tamper_gate(torch.cat((output.tamper_features, spectral), -1)))
        fused_aigc = aigc_gate * output.aigc_features + (1.0 - aigc_gate) * spectral
        fused_tamper = tamper_gate * output.tamper_features + (1.0 - tamper_gate) * spectral

        aigc_logit = self.heads.aigc_classifier(fused_aigc).squeeze(-1)
        tamper_logit = self.heads.tamper_classifier(fused_tamper).squeeze(-1)
        return ProvenanceOutput(
            aigc_logit=aigc_logit,
            tamper_logit=tamper_logit,
            probabilities=hierarchical_probabilities(aigc_logit, tamper_logit),
            aigc_features=fused_aigc,
            tamper_features=fused_tamper,
            token_tamper_logits=output.token_tamper_logits,
            fusion_gates=torch.cat((aigc_gate, tamper_gate), dim=-1),
        )

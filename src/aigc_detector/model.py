from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpointing
from PIL import Image
from torch import Tensor, nn


def hierarchical_probabilities(aigc_logit: Tensor, tamper_logit: Tensor) -> Tensor:
    """Return [authentic, tampered, fully-AIGC] probabilities from two decisions."""
    aigc = aigc_logit.sigmoid()
    tamper = tamper_logit.sigmoid()
    return torch.stack(((1.0 - aigc) * (1.0 - tamper), (1.0 - aigc) * tamper, aigc), dim=-1)


@dataclass
class ProvenanceOutput:
    aigc_logit: Tensor
    tamper_logit: Tensor
    probabilities: Tensor
    aigc_features: Tensor
    tamper_features: Tensor
    token_tamper_logits: list[Tensor]
    fusion_gates: Tensor | None = None

    @property
    def provenance_logits(self) -> Tensor:
        """Compatibility view for evaluators; softmax returns the model probabilities."""
        return self.probabilities.clamp_min(1e-7).log()

    @property
    def provenance_features(self) -> Tensor:
        return torch.cat((self.aigc_features, self.tamper_features), dim=-1)


class LearnedQueryPool(nn.Module):
    def __init__(self, encoder_dim: int, queries: int, heads: int = 4) -> None:
        super().__init__()
        if encoder_dim % heads:
            raise ValueError("encoder_dim must be divisible by attention heads")
        self.queries = nn.Parameter(torch.empty(queries, encoder_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(encoder_dim, heads, batch_first=True)

    def forward(self, tokens: Tensor) -> Tensor:
        query = self.queries.to(dtype=tokens.dtype).unsqueeze(0)
        pooled, _ = self.attention(
            query, tokens.unsqueeze(0), tokens.unsqueeze(0), need_weights=False
        )
        return pooled.squeeze(0)


class TaskSpecificProvenanceHead(nn.Module):
    """Independent global-AIGC and localized-tamper experts over Gemma tokens."""

    def __init__(
        self,
        encoder_dim: int = 1152,
        branch_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aigc_queries = LearnedQueryPool(encoder_dim, queries=4)
        self.tamper_query = LearnedQueryPool(encoder_dim, queries=1)
        self.token_tamper_classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, 1)
        )
        self.aigc_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim * 6),
            nn.Linear(encoder_dim * 6, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tamper_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim * 3),
            nn.Linear(encoder_dim * 3, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aigc_classifier = nn.Linear(branch_dim, 1)
        self.tamper_classifier = nn.Linear(branch_dim, 1)

    def extract_features(
        self, token_sequences: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        if not token_sequences:
            raise ValueError("At least one image token sequence is required")
        aigc_features: list[Tensor] = []
        tamper_features: list[Tensor] = []
        patch_logits: list[Tensor] = []
        for raw_tokens in token_sequences:
            if raw_tokens.ndim != 2:
                raise ValueError(f"Expected [tokens, hidden], got {tuple(raw_tokens.shape)}")
            tokens = raw_tokens.float()
            query_tokens = self.aigc_queries(tokens).flatten()
            aigc_input = torch.cat((query_tokens, tokens.mean(0), tokens.std(0, unbiased=False)))
            aigc_features.append(self.aigc_projection(aigc_input))

            scores = self.token_tamper_classifier(tokens).squeeze(-1)
            keep = max(1, ceil(tokens.shape[0] * 0.05))
            top_average = tokens[scores.topk(keep).indices].mean(0)
            attention_average = torch.sum(tokens * scores.softmax(0).unsqueeze(-1), dim=0)
            global_query = self.tamper_query(tokens).squeeze(0)
            tamper_features.append(
                self.tamper_projection(torch.cat((top_average, attention_average, global_query)))
            )
            patch_logits.append(scores)
        return torch.stack(aigc_features), torch.stack(tamper_features), patch_logits

    def forward(self, token_sequences: Sequence[Tensor]) -> ProvenanceOutput:
        aigc_features, tamper_features, token_logits = self.extract_features(token_sequences)
        aigc_logit = self.aigc_classifier(aigc_features).squeeze(-1)
        tamper_logit = self.tamper_classifier(tamper_features).squeeze(-1)
        return ProvenanceOutput(
            aigc_logit=aigc_logit,
            tamper_logit=tamper_logit,
            probabilities=hierarchical_probabilities(aigc_logit, tamper_logit),
            aigc_features=aigc_features,
            tamper_features=tamper_features,
            token_tamper_logits=token_logits,
        )


class SpectralExpert(nn.Module):
    """ConvNeXt expert over RGB + fixed residuals, augmented with radial FFT energy."""

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
        self.fft_projection = nn.Sequential(
            nn.Linear(fft_bins, 128), nn.GELU(), nn.Linear(128, output_dim)
        )
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
        arrays = []
        for image in images:
            resized = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            data = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
            arrays.append(data.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1))
        return torch.stack(arrays).to(device=device, dtype=torch.float32).div_(255.0)

    def _radial_fft(self, rgb: Tensor) -> Tensor:
        gray = rgb.mean(dim=1)
        spectrum = torch.log1p(torch.fft.rfft2(gray, norm="ortho").abs())
        spectrum = torch.fft.fftshift(spectrum, dim=(-2,))
        height, width = spectrum.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, device=rgb.device)
        x = torch.linspace(0.0, 1.0, width, device=rgb.device)
        radius = torch.sqrt(y[:, None].square() + x[None, :].square()).clamp_max(1.0)
        indices = torch.clamp((radius * self.fft_bins).long(), max=self.fft_bins - 1)
        vectors = []
        for index in range(self.fft_bins):
            mask = indices == index
            vectors.append(spectrum[:, mask].mean(dim=1))
        return torch.stack(vectors, dim=-1)

    def forward(self, images: Sequence[Image.Image]) -> Tensor:
        device = next(self.parameters()).device
        rgb = self._images_to_tensor(images, device)
        gray = rgb.mean(dim=1, keepdim=True)
        residuals = F.conv2d(gray, self.residual_kernels, padding=1)
        residuals = residuals.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        spatial = self.network(torch.cat((rgb, residuals), dim=1))
        return spatial + self.fft_projection(self._radial_fft(rgb))


class ProvenanceHead(TaskSpecificProvenanceHead):
    """Backwards-compatible name for the new task-specific Gemma head."""

    def __init__(
        self,
        encoder_dim: int = 1152,
        trunk_dim: int = 512,
        branch_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        del trunk_dim
        super().__init__(encoder_dim=encoder_dim, branch_dim=branch_dim, dropout=dropout)


class Gemma4VisionBackbone(nn.Module):
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
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Gemma 4 requires transformers==5.10.1") from exc
        self.visual_tokens = visual_tokens
        self.processor = Gemma4ImageProcessor.from_pretrained(encoder_id, revision=revision)
        self.encoder = Gemma4VisionModel.from_pretrained(encoder_id, revision=revision, dtype=dtype)
        self.set_frozen(freeze)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def set_frozen(self, frozen: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen

    def set_trainable_last_layers(self, count: int) -> None:
        layers = self.encoder.encoder.layers
        if not 0 <= count <= len(layers):
            raise ValueError(f"Requested {count} trainable layers; encoder has {len(layers)}")
        self.set_frozen(True)
        for layer in layers[len(layers) - count :] if count else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def forward(self, images: Sequence[Image.Image]) -> list[Tensor]:
        processed = self.processor(
            images=list(images), max_soft_tokens=self.visual_tokens, return_tensors="pt"
        )
        counts = [int(value) for value in processed["num_soft_tokens_per_image"].reshape(-1)]
        pixel_values = processed["pixel_values"].to(self.device)
        position_ids = processed["image_position_ids"].to(self.device)
        frozen = not any(parameter.requires_grad for parameter in self.encoder.parameters())
        context = torch.no_grad() if frozen else nullcontext()
        with context:
            output = self.encoder(pixel_values=pixel_values, pixel_position_ids=position_ids)
        hidden = output.last_hidden_state
        if sum(counts) != hidden.shape[0]:
            raise RuntimeError(f"Processor reported {sum(counts)} tokens, got {hidden.shape[0]}")
        return list(hidden.split(counts, dim=0))


class DINOv3VisionBackbone(nn.Module):
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
        except ImportError as exc:  # pragma: no cover
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
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen

    def set_trainable_last_layers(self, count: int) -> None:
        layers = self.encoder.model.layer
        if not 0 <= count <= len(layers):
            raise ValueError(f"Requested {count} trainable layers; encoder has {len(layers)}")
        self.set_frozen(True)
        for layer in layers[len(layers) - count :] if count else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        self.encoder.gradient_checkpointing_enable()

    def forward(self, images: Sequence[Image.Image]) -> list[Tensor]:
        pixels = self.processor(images=list(images), return_tensors="pt")["pixel_values"].to(
            self.device
        )
        frozen = not any(parameter.requires_grad for parameter in self.encoder.parameters())
        context = torch.no_grad() if frozen else nullcontext()
        with context:
            hidden = self.encoder(pixel_values=pixels).last_hidden_state[:, 5:]
        return list(hidden)


class PESpatialVisionBackbone(nn.Module):
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
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PE-Spatial requires perception_models and huggingface-hub") from exc
        weights = hf_hub_download(
            encoder_id,
            filename="PE-Spatial-G14-448.pt",
            revision=revision,
        )
        self.encoder = VisionTransformer.from_config(
            "PE-Spatial-G14-448",
            pretrained=True,
            checkpoint_path=weights,
            image_size=image_size,
        )
        self.encoder.to(dtype=dtype)
        self.image_size = image_size
        self.set_frozen(freeze)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def set_frozen(self, frozen: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen

    def set_trainable_last_layers(self, count: int) -> None:
        layers = self.encoder.transformer.resblocks
        if not 0 <= count <= len(layers):
            raise ValueError(f"Requested {count} trainable layers; encoder has {len(layers)}")
        self.set_frozen(True)
        for layer in layers[len(layers) - count :] if count else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        transformer = self.encoder.transformer
        transformer.grad_checkpointing = True

        def checkpointed_forward(
            hidden: Tensor, attention_mask: Tensor | None = None, layer_idx: int = -1
        ) -> Tensor:
            stop_idx = (transformer.layers + layer_idx) % transformer.layers
            for index, block in enumerate(transformer.resblocks):
                hidden = checkpointing.checkpoint(
                    block,
                    hidden,
                    attention_mask,
                    use_reentrant=False,
                )
                if index == stop_idx:
                    break
            return hidden

        transformer.forward = checkpointed_forward

    def forward(self, images: Sequence[Image.Image]) -> list[Tensor]:
        arrays = []
        for image in images:
            resized = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            data = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
            arrays.append(data.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1))
        pixels = torch.stack(arrays).to(
            device=self.device, dtype=next(self.encoder.parameters()).dtype
        )
        pixels = pixels.div_(255.0).sub_(0.5).div_(0.5)
        frozen = not any(parameter.requires_grad for parameter in self.encoder.parameters())
        context = torch.no_grad() if frozen else nullcontext()
        with context:
            hidden = self.encoder.forward_features(pixels, norm=False)
        return list(hidden)


def build_backbone(
    backbone_type: str,
    encoder_id: str,
    *,
    encoder_revision: str | None,
    visual_tokens: int,
    image_size: int,
    freeze: bool,
) -> nn.Module:
    if backbone_type == "gemma4":
        return Gemma4VisionBackbone(
            encoder_id,
            revision=encoder_revision,
            visual_tokens=visual_tokens,
            freeze=freeze,
        )
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
    raise ValueError(f"Unsupported backbone_type: {backbone_type}")


class ProvenanceModel(nn.Module):
    def __init__(
        self,
        encoder_id: str,
        *,
        encoder_revision: str | None,
        backbone_type: str = "gemma4",
        visual_tokens: int = 1120,
        image_size: int = 384,
        encoder_dim: int = 1152,
        trunk_dim: int = 512,
        branch_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        spectral_expert: bool = False,
        spectral_image_size: int = 384,
        spectral_pretrained: bool = False,
        use_token_adapter: bool = False,
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
        self.token_adapter = (
            nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, trunk_dim),
            )
            if use_token_adapter
            else nn.Identity()
        )
        head_dim = trunk_dim if use_token_adapter else encoder_dim
        self.heads = ProvenanceHead(
            encoder_dim=head_dim,
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
        adapted = [self.token_adapter(tokens.float()) for tokens in token_sequences]
        return self.heads(adapted)

    def forward(self, images: Sequence[Image.Image]) -> ProvenanceOutput:
        output = self.forward_tokens(self.backbone(images))
        if self.spectral is None:
            return output
        spectral = self.spectral(images)
        aigc_gate = torch.sigmoid(self.aigc_gate(torch.cat((output.aigc_features, spectral), -1)))
        tamper_gate = torch.sigmoid(
            self.tamper_gate(torch.cat((output.tamper_features, spectral), -1))
        )
        aigc_features = aigc_gate * output.aigc_features + (1.0 - aigc_gate) * spectral
        tamper_features = tamper_gate * output.tamper_features + (1.0 - tamper_gate) * spectral
        aigc_logit = self.heads.aigc_classifier(aigc_features).squeeze(-1)
        tamper_logit = self.heads.tamper_classifier(tamper_features).squeeze(-1)
        return ProvenanceOutput(
            aigc_logit=aigc_logit,
            tamper_logit=tamper_logit,
            probabilities=hierarchical_probabilities(aigc_logit, tamper_logit),
            aigc_features=aigc_features,
            tamper_features=tamper_features,
            token_tamper_logits=output.token_tamper_logits,
            fusion_gates=torch.cat((aigc_gate, tamper_gate), dim=-1),
        )

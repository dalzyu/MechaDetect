from __future__ import annotations

"""Training and adaptation pipeline for robust image provenance detection.

Key features:
1. Dual Data Modes:
   - Live Dataset: `PairedImageDataset` applies on-the-fly geometric preprocessing
     and synthetic transform augmentations (JPEG, blur, resize, noise, crop).
   - Cached Features: `CachedFeatureDataset` loads pre-extracted backbone tokens
     from disk, accelerating frozen-screening epochs by 10-20x.
2. Generator-Balanced Stratified Sampling:
   - Balances authentic negatives against fully generated and AI-edited positives.
   - Weights individual generators inversely by frequency to avoid dominance.
3. Layer-Wise Learning Rate Decay (LLRD):
   - Decays learning rate geometrically from top layers to bottom layers (decay = 0.85).
   - Task heads train at a higher learning rate, while deep backbone layers adapt
     conservatively to preserve pre-trained feature stability.
4. Binary AI-positive supervised loss.
5. Optional localized edit-mask supervision.
6. Transformation consistency and EMA teacher distillation.
"""

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from random import Random
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from .config import load_config
from .constants import Provenance, Transformation
from .adaptation import (
    apply_attention_lora,
    load_trainable_encoder_state,
    trainable_encoder_state,
)
from .dataset import PairedImageDataset, collate_pairs, load_manifest_frame, verify_materialization
from .ema import ParameterEMA
from .feature_cache import CachedFeatureDataset, collate_cached_features
from .losses import LossWeights, provenance_robustness_loss
from .manifests import assert_forbidden_demonstration_data_absent
from .model import ProvenanceModel, ProvenanceOutput, ai_generated_probability
from .preprocessing import mask_to_token_occupancy
from .runtime import (
    cleanup_distributed,
    is_main_process,
    load_local_environment,
    resolve_project_path,
    seed_everything,
    setup_distributed,
)
from .sampling import build_balanced_sampler


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a batch dictionary to the specified torch device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _loss_weights(config: dict[str, Any]) -> LossWeights:
    """Extract loss component weights from the configuration mapping."""
    values = config["loss"]
    return LossWeights(
        provenance_original=values["provenance_original"],
        provenance_transformed=values["provenance_transformed"],
        prediction_consistency=values["prediction_consistency"],
        feature_consistency=values.get("feature_consistency", 0.0),
        mask_focal=values.get("mask_focal", 0.0),
        mask_dice=values.get("mask_dice", 0.0),
        ema_consistency=values.get("ema_consistency", 0.0),
    )


def build_model(config: dict[str, Any]) -> ProvenanceModel:
    """Instantiate a ProvenanceModel configured for screening, adaptation, or full training."""
    values = config["model"]
    dtype_name = str(values.get("encoder_dtype", "float32")).lower()
    encoder_dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in encoder_dtypes:
        raise ValueError(
            f"Unsupported model.encoder_dtype {dtype_name!r}; use float32 or bfloat16"
        )
    model = ProvenanceModel(
        values["encoder_id"],
        encoder_revision=values.get("encoder_revision"),
        backbone_type=values.get("backbone_type", "gemma4"),
        visual_tokens=values.get("visual_tokens", 1120),
        image_size=values.get("image_size", 384),
        encoder_dim=values["encoder_dim"],
        trunk_dim=values["trunk_dim"],
        branch_dim=values["branch_dim"],
        dropout=values["dropout"],
        freeze_encoder=values["freeze_encoder"],
        spectral_expert=values.get("spectral_expert", False),
        spectral_image_size=values.get("spectral_image_size", 384),
        spectral_pretrained=values.get("spectral_pretrained", False),
        use_token_adapter=values.get("use_token_adapter", True),
        encoder_dtype=encoder_dtypes[dtype_name],
    )

    # Controlled backbone adaptation: unfreeze the final N transformer blocks
    trainable_last_layers = int(values.get("trainable_last_layers", 0))
    if trainable_last_layers:
        model.backbone.set_trainable_last_layers(trainable_last_layers)

    # Activation checkpointing saves memory whenever any backbone parameters are trainable.
    if values.get("gradient_checkpointing") and any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    ):
        model.backbone.enable_gradient_checkpointing()

    # Optional parameter-efficient fine-tuning via attention LoRA
    lora = values.get("lora", {})
    if lora.get("enabled"):
        model.backbone.set_frozen(True)
        apply_attention_lora(
            model.backbone.encoder,
            rank=int(lora.get("rank", 8)),
            alpha=float(lora.get("alpha", 16)),
            dropout=float(lora.get("dropout", 0.05)),
        )
        if values.get("gradient_checkpointing"):
            model.backbone.enable_gradient_checkpointing()

    return model


def _encoder_layer_index(name: str) -> int | None:
    """Parse the zero-indexed layer depth from parameter names across different ViT architectures."""
    for marker in (".layers.", ".resblocks.", ".layer."):
        if marker in name:
            value = name.split(marker, 1)[1].split(".", 1)[0]
            return int(value)
    return None


def build_optimizer(model: ProvenanceModel, config: dict[str, Any]) -> Optimizer:
    """Build AdamW with separate task-head LR and backbone layer-wise LR decay."""
    training = config["training"]
    encoder_parameters: list[tuple[int | None, nn.Parameter]] = []
    task_parameters: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            encoder_parameters.append((_encoder_layer_index(name), parameter))
        else:
            task_parameters.append(parameter)

    groups: list[dict[str, Any]] = [{"params": task_parameters, "lr": training["heads_lr"]}]

    if encoder_parameters:
        decay = float(training.get("layerwise_lr_decay", 1.0))
        layer_indices = [index for index, _ in encoder_parameters if index is not None]
        last_layer = max(layer_indices) if layer_indices else 0
        by_depth: dict[int, list[nn.Parameter]] = {}
        for index, parameter in encoder_parameters:
            # Patch/position embeddings have no transformer-layer index. They
            # are the earliest features, so give them the LOWEST LR rather
            # than accidentally treating them like the final layer.
            depth = last_layer + 1 if index is None else last_layer - index
            by_depth.setdefault(depth, []).append(parameter)

        groups.extend(
            {
                "params": parameters,
                "lr": float(training["encoder_lr"]) * (decay**depth),
            }
            for depth, parameters in sorted(by_depth.items())
        )

    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(
            f"Unsupported training.optimizer {optimizer_name!r}; use adamw"
        )
    return AdamW(groups, weight_decay=training["weight_decay"])


def build_scheduler(
    optimizer: Optimizer, total_updates: int, warmup_fraction: float
) -> LambdaLR:
    """Cosine learning rate decay schedule with linear warmup."""
    warmup_steps = max(1, round(total_updates * warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, schedule)


def save_checkpoint(
    model: ProvenanceModel,
    config: dict[str, Any],
    output_dir: Path,
    step: int,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    manifest_path: Path | None = None,
    ema: ParameterEMA | None = None,
    epoch: int = 0,
    micro_step: int = 0,
    *,
    filename: str | None = None,
    selection: dict[str, Any] | None = None,
) -> Path:
    """Serialize model weights and resumable state atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_trainable = any(p.requires_grad for p in model.backbone.parameters())
    payload: dict[str, Any] = {
        "step": step,
        "config": config,
        "heads": model.heads.state_dict(),
        "token_adapter": model.token_adapter.state_dict(),
        "encoder_trainable": encoder_trainable,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "epoch": epoch,
        "micro_step": micro_step,
    }
    if model.spectral is not None:
        payload["spectral"] = model.spectral.state_dict()
        payload["aigc_gate"] = model.aigc_gate.state_dict()
        payload["tamper_gate"] = model.tamper_gate.state_dict()
    if encoder_trainable:
        payload["encoder_trainable_state"] = trainable_encoder_state(model.backbone.encoder)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if manifest_path is not None:
        import hashlib
        payload["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if selection is not None:
        payload["selection"] = selection
    path = output_dir / (filename or f"checkpoint-step-{step}.pt")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
    return path


def cpu_rng_states(states: list[torch.Tensor]) -> list[torch.Tensor]:
    """Ensure CUDA RNG states are properly formatted as CPU ByteTensors."""
    normalized = []
    for state in states:
        if not isinstance(state, torch.Tensor) or state.dtype is not torch.uint8:
            raise TypeError("CUDA RNG checkpoint state must be a torch.ByteTensor")
        normalized.append(state.detach().cpu().contiguous())
    return normalized


def restore_checkpoint(
    path: Path,
    model: ProvenanceModel,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    ema: ParameterEMA | None,
    manifest_path: Path,
    device: torch.device,
) -> tuple[int, int, int]:
    """Restore training state from a previously saved checkpoint."""
    import hashlib

    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if payload.get("manifest_sha256") != expected_manifest:
        raise RuntimeError("Resume checkpoint was created from a different training manifest")

    model.heads.load_state_dict(payload["heads"])
    adapter_state = payload.get("token_adapter", {})
    if model.token_adapter.state_dict() or adapter_state:
        model.token_adapter.load_state_dict(adapter_state)

    if model.spectral is not None:
        model.spectral.load_state_dict(payload["spectral"])
        model.aigc_gate.load_state_dict(payload["aigc_gate"])
        model.tamper_gate.load_state_dict(payload["tamper_gate"])

    if payload.get("encoder_trainable"):
        load_trainable_encoder_state(model.backbone.encoder, payload["encoder_trainable_state"])

    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])

    if ema is not None:
        if "ema" not in payload:
            raise RuntimeError("EMA is enabled in config but checkpoint has no EMA state")
        ema.load_state_dict(payload["ema"])

    torch.set_rng_state(payload["rng_cpu"].cpu())
    if torch.cuda.is_available() and payload.get("rng_cuda"):
        torch.cuda.set_rng_state_all(cpu_rng_states(payload["rng_cuda"]))

    return int(payload["step"]), int(payload.get("epoch", 0)), int(payload.get("micro_step", 0))


def _compute_micro_step_loss(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    weights: LossWeights,
    is_cached: bool,
    clean_only: bool,
    ema: ParameterEMA | None,
    update_step: int,
    ema_start_step: int,
    confidence_threshold: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute model predictions and losses for one small batch.

    ``model`` may be a normal ProvenanceModel or a DDP wrapper. The student
    forward must use the wrapper so gradients synchronise. EMA teacher
    inference uses the underlying model because it has no gradients to sync.
    """
    raw_model = model.module if isinstance(model, DDP) else model
    teacher_probabilities = None

    if is_cached:
        original_tokens = [t.to(device) for t in batch["original_tokens"]]
        transformed_tokens = [t.to(device) for t in batch["transformed_tokens"]]

        if ema is not None and update_step >= ema_start_step:
            with ema.average_parameters(raw_model), torch.no_grad():
                teacher_probabilities = raw_model.forward_tokens(
                    original_tokens
                ).probabilities.detach()

        original = raw_model.forward_tokens(original_tokens)
        transformed = (
            original if clean_only else raw_model.forward_tokens(transformed_tokens)
        )
        token_mask_targets = [
            None if target is None else target.to(device)
            for target in batch["token_mask_targets"]
        ]
    else:
        if ema is not None and update_step >= ema_start_step:
            teacher_probabilities = ema.forward(
                raw_model, batch["original"]
            ).probabilities.detach()

        if clean_only:
            original = model(batch["original"])
            transformed = original
        else:
            # One DDP forward per micro-step. Two forwards before one backward
            # can confuse DDP's reducer, and concatenation has the same retained
            # activation footprint as two separately retained graphs.
            original_count = len(batch["original"])
            combined = model([*batch["original"], *batch["transformed"]])
            original = ProvenanceOutput(
                ai_positive_logit=combined.ai_positive_logit[:original_count],
                probabilities=combined.probabilities[:original_count],
                aigc_features=combined.aigc_features[:original_count],
                tamper_features=combined.tamper_features[:original_count],
                token_tamper_logits=combined.token_tamper_logits[:original_count],
                fusion_gates=(
                    None
                    if combined.fusion_gates is None
                    else combined.fusion_gates[:original_count]
                ),
            )
            transformed = ProvenanceOutput(
                ai_positive_logit=combined.ai_positive_logit[original_count:],
                probabilities=combined.probabilities[original_count:],
                aigc_features=combined.aigc_features[original_count:],
                tamper_features=combined.tamper_features[original_count:],
                token_tamper_logits=combined.token_tamper_logits[original_count:],
                fusion_gates=(
                    None
                    if combined.fusion_gates is None
                    else combined.fusion_gates[original_count:]
                ),
            )
        token_mask_targets = [
            None
            if mask is None
            else mask_to_token_occupancy(mask, logits.numel(), image.size)
            for mask, logits, image in zip(
                batch["mask"],
                original.token_tamper_logits,
                batch["original"],
                strict=True,
            )
        ]

    ai_target = batch.get("ai_positive")
    if ai_target is None and "provenance" in batch:
        ai_target = (batch["provenance"] != int(Provenance.AUTHENTIC)).float()

    loss, components = provenance_robustness_loss(
        original,
        transformed,
        provenance=batch.get("provenance"),
        ai_target=ai_target,
        weights=weights,
        token_mask_targets=token_mask_targets,
        teacher_probabilities=teacher_probabilities,
        teacher_confidence_threshold=confidence_threshold,
    )
    return loss, components


@torch.inference_mode()
def _validate(
    model: ProvenanceModel,
    val_manifest: Path,
    config: dict[str, Any],
    device: torch.device,
    data_root: Path | None,
    max_rows: int | None = None,
) -> dict[str, float]:
    """Run deterministic clean validation on a bounded probe."""
    from .metrics import balanced_accuracy, binary_auroc, confusion_matrix, macro_f1

    dataset = PairedImageDataset(
        val_manifest,
        data_root=data_root,
        seed=int(config["seed"]),
        transform_families=(),
        render_policy=config.get("preprocessing", {}).get(
            "policy", "square_jpeg95"
        ),
    )
    if max_rows is not None:
        dataset = Subset(dataset, range(min(int(max_rows), len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("validation_batch_size", 4)),
        shuffle=False,
        num_workers=int(config["training"].get("validation_workers", 2)),
        collate_fn=collate_pairs,
        pin_memory=True,
    )

    from .constants import Transformation
    from .transforms import TransformSpec, apply_transform

    probe = {
        "clean": None,
        "jpeg50": TransformSpec(Transformation.JPEG, 50.0),
        "blur1": TransformSpec(Transformation.BLUR, 1.0),
    }
    targets: list[Tensor] = []
    probabilities: dict[str, list[Tensor]] = {name: [] for name in probe}
    was_training = model.training
    model.eval()
    precision = config["training"]["precision"]
    for batch_index, raw_batch in enumerate(loader):
        batch = _to_device(raw_batch, device)
        targets.append(batch["ai_positive"].cpu().long())
        for name, transform in probe.items():
            images = (
                batch["original"]
                if transform is None
                else [
                    apply_transform(image, transform, Random(10_000 + batch_index * 10 + offset))
                    for offset, image in enumerate(batch["original"])
                ]
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            )
            with autocast:
                output = model(images)
            probabilities[name].append(output.probabilities.float().cpu())
    model.train(was_training)

    ai_target = torch.cat(targets).long()
    condition_metrics: dict[str, dict[str, float]] = {}
    for name, values in probabilities.items():
        probability = torch.cat(values)
        ai_score = ai_generated_probability(probability)
        ai_prediction = (ai_score >= 0.5).long()
        matrix = confusion_matrix(ai_target, ai_prediction, classes=2)
        condition_metrics[name] = {
            "ai_positive_auroc": binary_auroc(ai_target.float(), ai_score),
            "ai_positive_recall": float(matrix[1, 1] / matrix[1].sum()) if matrix[1].sum() else 0.0,
            "authentic_recall": float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else 0.0,
        }
    clean_probability = torch.cat(probabilities["clean"])
    clean_score = ai_generated_probability(clean_probability)
    clean_prediction = (clean_score >= 0.5).long()
    clean_matrix = confusion_matrix(ai_target, clean_prediction, classes=2)
    clean_auroc = condition_metrics["clean"]["ai_positive_auroc"]
    return {
        "accuracy": float((clean_prediction == ai_target).float().mean()),
        "balanced_accuracy": balanced_accuracy(clean_matrix),
        "macro_f1": macro_f1(clean_matrix),
        "ai_positive_auroc": clean_auroc,
        "ai_positive_recall": condition_metrics["clean"]["ai_positive_recall"],
        "authentic_recall": condition_metrics["clean"]["authentic_recall"],
        "robust_probe_mean_auroc": sum(
            metric["ai_positive_auroc"] for metric in condition_metrics.values()
        ) / len(condition_metrics),
    }


def run_training(
    config_path: Path,
    max_steps: int | None = None,
    train_manifest_override: Path | None = None,
    resume_path: Path | None = None,
    initial_checkpoint_override: Path | None = None,
    render_policy_override: str | None = None,
    stage_override: str | None = None,
) -> Path:
    """Train the teacher on one GPU or many GPUs.

    Plain ``python -m aigc_detector.train`` uses one GPU. Launching the same
    command through ``torchrun`` enables DistributedDataParallel (DDP): every
    GPU owns a model copy, processes a different data shard, and averages
    gradients at optimizer-step boundaries.
    """
    project_root = config_path.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(config_path)

    if render_policy_override is not None:
        config.setdefault("preprocessing", {})["policy"] = render_policy_override
    if stage_override is not None:
        config["training"]["stage"] = stage_override

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for teacher training")
    # CMP 170HX exposes Tensor Cores through BF16/TF32.  Keep the model's
    # master weights in FP32 while allowing matmul/convolution kernels to use
    # the accelerated paths selected by the training precision.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, local_rank = setup_distributed()
    required_world_size = config["training"].get("required_world_size")
    if required_world_size is not None and world_size != int(required_world_size):
        cleanup_distributed()
        raise RuntimeError(
            f"Expected exactly {int(required_world_size)} training processes; got {world_size}"
        )
    device = torch.device(f"cuda:{local_rank}")

    # All ranks start from identical model weights. Their data differs because
    # the sampler below gives each rank a separate shard.
    weights = _loss_weights(config)
    clean_only = not bool(config.get("transforms", {}).get("families"))
    seed_everything(int(config["seed"]))
    training = config["training"]
    manifest = train_manifest_override or resolve_project_path(
        config["paths"]["train_manifest"], project_root
    )

    assert_forbidden_demonstration_data_absent(load_manifest_frame(manifest))
    require_mat = bool(config["paths"].get("require_materialized", True))
    verify_materialization(
        manifest,
        data_root=config["paths"].get("data_root"),
        allow_missing=not require_mat,
    )

    # Cached features are a single-GPU bake-off path.  Production teacher
    # training intentionally runs live images on every DDP rank so the spectral
    # branch and paired transforms remain in the trained graph.
    cache_root = config["paths"].get("feature_cache")
    is_cached = bool(cache_root)
    if is_cached and world_size > 1:
        cleanup_distributed()
        raise RuntimeError(
            "Cached feature training is single-GPU only. Remove paths.feature_cache "
            "for multi-GPU teacher training."
        )

    if is_cached:
        dataset = CachedFeatureDataset(
            manifest, resolve_project_path(cache_root, project_root)
        )
        collate_fn = collate_cached_features
    else:
        dataset = PairedImageDataset(
            manifest,
            data_root=config["paths"]["data_root"],
            seed=int(config["seed"]),
            transform_families=tuple(
                Transformation[name.upper()]
                for name in config["transforms"].get("families", [])
            ),
            render_policy=config.get("preprocessing", {}).get(
                "policy", "square_jpeg95"
            ),
            runtime_fetch=bool(config["paths"].get("runtime_fetch", True)),
            allow_missing=not require_mat,
        )
        collate_fn = collate_pairs

    use_balanced_sampler = bool(training.get("generator_balanced_sampler", True))
    sampler = (
        build_balanced_sampler(
            [int(record.provenance) for record in dataset.records],
            [record.generator for record in dataset.records],
            samples=len(dataset),
            seed=int(config["seed"]),
            rank=rank,
            world_size=world_size,
            ai_positive=[int(record.ai_positive) for record in dataset.records],
            max_ratio=float(training.get("sampler_max_ratio", 5.0)),
        )
        if use_balanced_sampler
        else None
    )
    if world_size > 1 and sampler is None:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config["seed"]),
        )

    loader = DataLoader(
        dataset,
        batch_size=int(training["physical_batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=int(training["num_workers"]) > 0,
        generator=torch.Generator().manual_seed(int(config["seed"]) + 10_000),
    )

    # Build and load the ordinary model first. DDP wraps this same object later;
    # raw_model remains the convenient handle for EMA and checkpoint code.
    raw_model = build_model(config)
    if is_cached:
        raw_model.backbone.to("cpu")
        raw_model.token_adapter.to(device)
        raw_model.heads.to(device)
    else:
        raw_model.to(device)

    initial_checkpoint = initial_checkpoint_override or config["paths"].get(
        "initial_checkpoint"
    )
    initial_ema_state = None
    if initial_checkpoint and resume_path is None:
        payload = torch.load(
            resolve_project_path(initial_checkpoint, project_root),
            map_location=device,
            weights_only=False,
        )
        adapter_state = payload.get("token_adapter", {})
        if raw_model.token_adapter.state_dict() or adapter_state:
            raw_model.token_adapter.load_state_dict(adapter_state)
        raw_model.heads.load_state_dict(payload["heads"])
        if payload.get("encoder_trainable") and "encoder_trainable_state" in payload:
            load_trainable_encoder_state(
                raw_model.backbone.encoder, payload["encoder_trainable_state"]
            )
        if raw_model.spectral is not None:
            raw_model.spectral.load_state_dict(payload["spectral"])
            raw_model.aigc_gate.load_state_dict(payload["aigc_gate"])
            raw_model.tamper_gate.load_state_dict(payload["tamper_gate"])
        initial_ema_state = payload.get("ema")

    optimizer = build_optimizer(raw_model, config)
    if is_cached and not clean_only:
        cleanup_distributed()
        raise RuntimeError(
            "Cached features contain no transformed view. Set transformed and "
            "consistency loss weights to zero or remove paths.feature_cache."
        )

    accumulation = int(training["gradient_accumulation"])
    if accumulation <= 0:
        cleanup_distributed()
        raise ValueError("training.gradient_accumulation must be positive")
    micro_steps_per_epoch = len(loader)
    total_updates = (
        math.ceil(micro_steps_per_epoch / accumulation) * int(training["epochs"])
    )
    configured_budget = max_steps
    if configured_budget is None:
        configured_budget = training.get("max_updates")
    stop_step = (
        min(total_updates, int(configured_budget))
        if configured_budget is not None
        else total_updates
    )
    scheduler = build_scheduler(
        optimizer, stop_step, float(training["warmup_fraction"])
    )

    ema_config = training.get("ema", {})
    ema = (
        ParameterEMA(raw_model, decay=float(ema_config.get("decay", 0.999)))
        if ema_config.get("enabled")
        else None
    )
    if ema is not None and initial_ema_state is not None:
        ema.load_state_dict(initial_ema_state)
    ema_start = int(ema_config.get("start_update", 0))
    ema_threshold = float(ema_config.get("confidence_threshold", 0.8))

    update_step = 0
    start_epoch = 0
    resume_micro_step = 0
    if resume_path is not None:
        update_step, start_epoch, resume_micro_step = restore_checkpoint(
            resume_path, raw_model, optimizer, scheduler, ema, manifest, device
        )
        if resume_micro_step >= micro_steps_per_epoch:
            start_epoch += 1
            resume_micro_step = 0
        if is_main_process():
            print(
                f"resumed={resume_path} update={update_step} epoch={start_epoch + 1} "
                f"micro_step={resume_micro_step}",
                flush=True,
            )
        if update_step >= stop_step:
            cleanup_distributed()
            return resume_path if rank == 0 else Path()

    model: nn.Module = raw_model
    if world_size > 1:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )

    output_dir = Path(config["paths"]["output_root"]) / training["stage"]
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved-config.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(config, handle, indent=2)
        print(
            f"training devices={world_size} per_gpu_batch={training['physical_batch_size']} "
            f"gradient_accumulation={accumulation}",
            flush=True,
        )
    if world_size > 1:
        torch.distributed.barrier()

    val_manifest_value = config["paths"].get("val_manifest")
    val_interval = int(training.get("validation_interval", 0))
    validation_data_root = (
        resolve_project_path(config["paths"]["data_root"], project_root)
        if config["paths"].get("data_root")
        else None
    )

    optimizer.zero_grad(set_to_none=True)
    window_loss = 0.0
    window_components: dict[str, float] = {}
    window_micro_steps = 0
    should_stop = False
    last_epoch = max(0, start_epoch)
    last_micro_step = 0
    model.train()

    best_metric = float("-inf")
    best_selection: dict[str, Any] | None = None
    for epoch in range(start_epoch, int(training["epochs"])):
        last_epoch = epoch
        dataset.set_epoch(epoch)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        for micro_step, raw_batch in enumerate(loader, start=1):
            last_micro_step = micro_step
            if epoch == start_epoch and micro_step <= resume_micro_step:
                continue

            batch = _to_device(raw_batch, device)
            is_boundary = (
                micro_step % accumulation == 0 or micro_step == micro_steps_per_epoch
            )
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and not is_boundary
                else nullcontext()
            )

            # Divide by the actual accumulation-window size. This matters for
            # the short final window when epoch length is not divisible by N.
            window_start = ((micro_step - 1) // accumulation) * accumulation + 1
            window_size = min(
                accumulation, micro_steps_per_epoch - window_start + 1
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if training["precision"] == "bf16"
                else nullcontext()
            )
            with sync_context:
                with autocast:
                    loss, components = _compute_micro_step_loss(
                        model=model,
                        batch=batch,
                        device=device,
                        weights=weights,
                        is_cached=is_cached,
                        clean_only=clean_only,
                        ema=ema,
                        update_step=update_step,
                        ema_start_step=ema_start,
                        confidence_threshold=ema_threshold,
                    )
                    scaled_loss = loss / window_size
                scaled_loss.backward()

            window_loss += loss.detach().item()
            for name, value in components.items():
                window_components[name] = (
                    window_components.get(name, 0.0) + value.detach().item()
                )
            window_micro_steps += 1

            if not is_boundary:
                continue

            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), training["gradient_clip_norm"]
            )
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(raw_model)
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

            if is_main_process():
                divisor = float(window_micro_steps)
                component_text = " ".join(
                    f"{name}={value / divisor:.4f}"
                    for name, value in window_components.items()
                )
                print(
                    f"epoch={epoch + 1} update={update_step}/{stop_step} "
                    f"loss={window_loss / divisor:.4f} {component_text}",
                    flush=True,
                )
            window_loss = 0.0
            window_components = {}
            window_micro_steps = 0

            checkpoint_interval = int(training.get("checkpoint_interval", 0))
            should_checkpoint = (
                checkpoint_interval > 0 and update_step % checkpoint_interval == 0
            )
            if should_checkpoint and is_main_process():
                saved = save_checkpoint(
                    raw_model,
                    config,
                    output_dir,
                    update_step,
                    optimizer,
                    scheduler,
                    manifest,
                    ema,
                    epoch,
                    micro_step,
                )
                print(f"checkpoint={saved}", flush=True)
            if should_checkpoint and world_size > 1:
                torch.distributed.barrier()

            should_validate = (
                val_interval > 0
                and val_manifest_value
                and update_step % val_interval == 0
            )
            if should_validate and is_main_process():
                validation_manifest = resolve_project_path(val_manifest_value, project_root)
                online_metrics = _validate(
                    raw_model,
                    validation_manifest,
                    config,
                    device,
                    data_root=validation_data_root,
                    max_rows=training.get("validation_rows"),
                )
                candidates = [(False, online_metrics)]
                if ema is not None:
                    with ema.average_parameters(raw_model):
                        candidates.append(
                            (
                                True,
                                _validate(
                                    raw_model,
                                    validation_manifest,
                                    config,
                                    device,
                                    data_root=validation_data_root,
                                    max_rows=training.get("validation_rows"),
                                ),
                            )
                        )
                def guarded_score(item: tuple[bool, dict[str, float]]) -> float:
                    candidate = item[1]
                    if (
                        candidate["ai_positive_recall"] < 0.60
                        or candidate["authentic_recall"] < 0.60
                    ):
                        return float("-inf")
                    return float(candidate["robust_probe_mean_auroc"])

                use_ema, metrics = max(candidates, key=guarded_score)
                metric_text = " ".join(
                    f"{name}={value:.4f}" for name, value in metrics.items()
                )
                print(
                    f"validation update={update_step} selected={'ema' if use_ema else 'online'} "
                    f"{metric_text}",
                    flush=True,
                )
                score = guarded_score((use_ema, metrics))
                if score > best_metric:
                    best_metric = score
                    best_selection = {
                        "metric": "robust_probe_mean_auroc",
                        "value": score,
                        "update": update_step,
                        "probe_rows": training.get("validation_rows"),
                        "use_ema": use_ema,
                        "online": online_metrics,
                    }
                    best_path = save_checkpoint(
                        raw_model,
                        config,
                        output_dir,
                        update_step,
                        optimizer,
                        scheduler,
                        manifest,
                        ema,
                        epoch,
                        micro_step,
                        filename="checkpoint-best.pt",
                        selection=best_selection,
                    )
                    print(f"best_checkpoint={best_path}", flush=True)
            if should_validate and world_size > 1:
                torch.distributed.barrier()

            if update_step >= stop_step:
                should_stop = True
                break

        if should_stop:
            break

    if world_size > 1:
        torch.distributed.barrier()
    checkpoint = (
        save_checkpoint(
            raw_model,
            config,
            output_dir,
            update_step,
            optimizer,
            scheduler,
            manifest,
            ema,
            last_epoch,
            last_micro_step,
        )
        if is_main_process()
        else Path()
    )
    if is_main_process() and best_metric == float("-inf"):
        save_checkpoint(
            raw_model,
            config,
            output_dir,
            update_step,
            optimizer,
            scheduler,
            manifest,
            ema,
            last_epoch,
            last_micro_step,
            filename="checkpoint-best.pt",
            selection={"metric": "final_state", "update": update_step},
        )
    if world_size > 1:
        torch.distributed.barrier()
    cleanup_distributed()
    return checkpoint


def main() -> None:
    """CLI entrypoint for running training via python -m aigc_detector.train."""
    parser = argparse.ArgumentParser(
        description="Train or adapt the robust provenance detection model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/teacher_dinov3_stage1_clean_frozen.yaml"),
    )
    parser.add_argument(
        "--max-steps", type=int, help="Optional update step limit (for fast testing)."
    )
    parser.add_argument("--train-manifest", type=Path, help="Override training manifest CSV path.")
    parser.add_argument("--resume", type=Path, help="Resume training from an existing checkpoint.")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Initialize model weights from an earlier stage without restoring optimizer state.",
    )
    parser.add_argument(
        "--render-policy",
        choices=["square_jpeg95", "aspect_jpeg95", "aspect_randomized"],
    )
    parser.add_argument(
        "--stage", type=str, help="Override output experiment stage directory name."
    )
    args = parser.parse_args()

    checkpoint = run_training(
        args.config,
        max_steps=args.max_steps,
        train_manifest_override=args.train_manifest,
        resume_path=args.resume,
        initial_checkpoint_override=args.initial_checkpoint,
        render_policy_override=args.render_policy,
        stage_override=args.stage,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Training complete. Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .adaptation import (
    apply_attention_lora,
    load_trainable_encoder_state,
    trainable_encoder_state,
)
from .config import load_config
from .dataset import PairedImageDataset, collate_pairs
from .feature_cache import CachedFeatureDataset, collate_cached_features
from .ema import ParameterEMA
from .losses import LossWeights, provenance_robustness_loss
from .model import ProvenanceModel
from .preprocessing import mask_to_token_occupancy
from .runtime import load_local_environment, resolve_project_path, seed_everything
from .sampling import EpochWeightedSampler, build_balanced_sampler


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _loss_weights(config: dict[str, Any]) -> LossWeights:
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
    values = config["model"]
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
    )
    trainable_last_layers = int(values.get("trainable_last_layers", 0))
    if trainable_last_layers:
        model.backbone.set_trainable_last_layers(trainable_last_layers)
    if values.get("gradient_checkpointing") and trainable_last_layers:
        model.backbone.enable_gradient_checkpointing()
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


def build_optimizer(model: ProvenanceModel, config: dict[str, Any]) -> AdamW:
    training = config["training"]
    encoder_parameters = []
    task_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder_parameters if name.startswith("backbone.") else task_parameters).append(parameter)
    groups: list[dict[str, Any]] = [{"params": task_parameters, "lr": training["heads_lr"]}]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": training["encoder_lr"]})
    return AdamW(groups, weight_decay=training["weight_decay"])


def build_scheduler(optimizer: AdamW, total_updates: int, warmup_fraction: float) -> LambdaLR:
    warmup = max(1, round(total_updates * warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1e-3, step / warmup)
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, schedule)


def save_checkpoint(
    model: ProvenanceModel,
    config: dict[str, Any],
    output_dir: Path,
    step: int,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    manifest_path: Path | None = None,
    ema: ParameterEMA | None = None,
    epoch: int = 0,
    micro_step: int = 0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_trainable = any(parameter.requires_grad for parameter in model.backbone.parameters())
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
    path = output_dir / f"checkpoint-step-{step}.pt"
    torch.save(payload, path)
    return path


def cpu_rng_states(states: list[torch.Tensor]) -> list[torch.Tensor]:
    normalized = []
    for state in states:
        if not isinstance(state, torch.Tensor) or state.dtype is not torch.uint8:
            raise TypeError("CUDA RNG checkpoint state must be a torch.ByteTensor")
        normalized.append(state.detach().cpu().contiguous())
    return normalized


def restore_checkpoint(
    path: Path,
    model: ProvenanceModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    ema: ParameterEMA | None,
    manifest_path: Path,
    device: torch.device,
) -> tuple[int, int, int]:
    import hashlib

    payload = torch.load(path, map_location=device, weights_only=False)
    expected_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if payload.get("manifest_sha256") != expected_manifest:
        raise RuntimeError("Resume checkpoint was created from a different training manifest")
    model.heads.load_state_dict(payload["heads"])
    model.token_adapter.load_state_dict(payload["token_adapter"])
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
            raise RuntimeError("EMA is enabled but the checkpoint has no EMA state")
        ema.load_state_dict(payload["ema"])
    torch.set_rng_state(payload["rng_cpu"].cpu())
    if torch.cuda.is_available() and payload.get("rng_cuda"):
        torch.cuda.set_rng_state_all(cpu_rng_states(payload["rng_cuda"]))
    return int(payload["step"]), int(payload.get("epoch", 0)), int(payload.get("micro_step", 0))


def run_training(
    config_path: Path,
    max_steps: int | None = None,
    train_manifest_override: Path | None = None,
    resume_path: Path | None = None,
    render_policy_override: str | None = None,
    stage_override: str | None = None,
) -> Path:
    project_root = config_path.resolve().parent.parent
    load_local_environment(project_root)
    config = load_config(config_path)
    if render_policy_override is not None:
        config.setdefault("preprocessing", {})["policy"] = render_policy_override
    if stage_override is not None:
        config["training"]["stage"] = stage_override
    seed_everything(int(config["seed"]))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for backbone training")
    device = torch.device("cuda")
    training = config["training"]
    manifest = train_manifest_override or resolve_project_path(
        config["paths"]["train_manifest"], project_root
    )
    cache_root = config["paths"].get("feature_cache")
    if cache_root:
        dataset = CachedFeatureDataset(manifest, resolve_project_path(cache_root, project_root))
        collate_fn = collate_cached_features
    else:
        dataset = PairedImageDataset(
            manifest,
            data_root=config["paths"]["data_root"],
            seed=int(config["seed"]),
            chain_length_probabilities={
                int(length): float(probability)
                for length, probability in config["transforms"]["chain_length_probabilities"].items()
            },
            render_policy=config.get("preprocessing", {}).get("policy", "square_jpeg95"),
        )
        collate_fn = collate_pairs
    use_balanced_sampler = bool(training.get("generator_balanced_sampler", True))
    sampler = (
        build_balanced_sampler(
            [int(record.provenance) for record in dataset.records],
            [record.generator for record in dataset.records],
            samples=len(dataset),
            seed=int(config["seed"]),
        )
        if use_balanced_sampler
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["physical_batch_size"]),
        shuffle=not use_balanced_sampler,
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=True,
        generator=torch.Generator().manual_seed(int(config["seed"]) + 10_000),
    )

    model = build_model(config)
    if cache_root:
        model.backbone.to("cpu")
        model.token_adapter.to(device)
        model.heads.to(device)
    else:
        model.to(device)
    optimizer = build_optimizer(model, config)
    weights = _loss_weights(config)
    accumulation = int(training["gradient_accumulation"])
    total_updates = math.ceil(len(loader) / accumulation) * int(training["epochs"])
    stop_step = min(total_updates, max_steps) if max_steps is not None else total_updates
    scheduler = build_scheduler(optimizer, total_updates, float(training["warmup_fraction"]))
    ema_config = training.get("ema", {})
    ema = (
        ParameterEMA(model, decay=float(ema_config.get("decay", 0.999)))
        if ema_config.get("enabled")
        else None
    )
    update_step = 0
    start_epoch = 0
    resume_micro_step = 0
    if resume_path is not None:
        update_step, start_epoch, resume_micro_step = restore_checkpoint(
            resume_path, model, optimizer, scheduler, ema, manifest, device
        )
        if resume_micro_step >= len(loader):
            start_epoch += 1
            resume_micro_step = 0
        print(
            f"resumed={resume_path} update={update_step} epoch={start_epoch + 1} "
            f"micro_step={resume_micro_step}",
            flush=True,
        )
        if update_step >= stop_step:
            return resume_path

    output_dir = Path(config["paths"]["output_root"]) / training["stage"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved-config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    optimizer.zero_grad(set_to_none=True)
    window_loss = 0.0
    window_components: dict[str, float] = {}
    window_micro_steps = 0
    model.train()
    for epoch in range(start_epoch, int(training["epochs"])):
        dataset.set_epoch(epoch)
        if isinstance(sampler, EpochWeightedSampler):
            sampler.set_epoch(epoch)
        for micro_step, raw_batch in enumerate(loader, start=1):
            if epoch == start_epoch and micro_step <= resume_micro_step:
                continue
            batch = _to_device(raw_batch, device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if training["precision"] == "bf16"
                else nullcontext()
            )
            with autocast:
                teacher_probabilities = None
                if cache_root:
                    original_tokens = [tokens.to(device) for tokens in batch["original_tokens"]]
                    transformed_tokens = [tokens.to(device) for tokens in batch["transformed_tokens"]]
                    if ema is not None and update_step >= int(ema_config.get("start_update", 0)):
                        with ema.average_parameters(model), torch.no_grad():
                            teacher_probabilities = model.forward_tokens(
                                original_tokens
                            ).probabilities.detach()
                    original = model.forward_tokens(original_tokens)
                    transformed = model.forward_tokens(transformed_tokens)
                    token_mask_targets = [
                        None if target is None else target.to(device)
                        for target in batch["token_mask_targets"]
                    ]
                else:
                    if ema is not None and update_step >= int(ema_config.get("start_update", 0)):
                        with ema.average_parameters(model), torch.no_grad():
                            teacher_probabilities = model(batch["original"]).probabilities.detach()
                    original = model(batch["original"])
                    transformed = model(batch["transformed"])
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
                loss, components = provenance_robustness_loss(
                    original,
                    transformed,
                    provenance=batch["provenance"],
                    weights=weights,
                    token_mask_targets=token_mask_targets,
                    teacher_probabilities=teacher_probabilities,
                    teacher_confidence_threshold=float(ema_config.get("confidence_threshold", 0.8)),
                )
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            window_loss += loss.detach().item()
            for name, value in components.items():
                window_components[name] = window_components.get(name, 0.0) + value.detach().item()
            window_micro_steps += 1

            if micro_step % accumulation == 0 or micro_step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                divisor = float(window_micro_steps)
                component_text = " ".join(
                    f"{name}={value / divisor:.4f}" for name, value in window_components.items()
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
                if checkpoint_interval and update_step % checkpoint_interval == 0:
                    saved = save_checkpoint(
                        model,
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
                if max_steps is not None and update_step >= stop_step:
                    return save_checkpoint(
                        model,
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
    return save_checkpoint(
        model,
        config,
        output_dir,
        update_step,
        optimizer,
        scheduler,
        manifest,
        ema,
        max(0, int(training["epochs"]) - 1),
        len(loader),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--render-policy",
        choices=["square_jpeg95", "aspect_jpeg95", "aspect_randomized"],
    )
    parser.add_argument("--stage")
    args = parser.parse_args()
    checkpoint = run_training(
        args.config,
        max_steps=args.max_steps,
        train_manifest_override=args.train_manifest,
        resume_path=args.resume,
        render_policy_override=args.render_policy,
        stage_override=args.stage,
    )
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()

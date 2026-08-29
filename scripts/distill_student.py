from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from aigc_detector.config import load_config
from aigc_detector.constants import Transformation
from aigc_detector.dataset import PairedImageDataset, collate_pairs, verify_materialization
from aigc_detector.losses import LossWeights, provenance_robustness_loss
from aigc_detector.model import ProvenanceOutput
from aigc_detector.predict import _load_checkpoint
from aigc_detector.runtime import (
    cleanup_distributed,
    is_main_process,
    load_local_environment,
    setup_distributed,
)
from aigc_detector.sampling import build_balanced_sampler
from aigc_detector.train import build_model, build_optimizer, build_scheduler, save_checkpoint

STUDENT_PRESETS = {
    "small": {
        "encoder_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "encoder_dim": 384,
        "max_updates": 1000,
        "epochs": 4,
    },
    "base": {
        "encoder_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "encoder_dim": 768,
        "max_updates": 1000,
        "epochs": 3,
    },
}


def student_config(teacher_config: dict, variant: str) -> dict:
    values = STUDENT_PRESETS[variant]
    config = copy.deepcopy(teacher_config)
    config["model"].update(
        {
            "backbone_type": "dinov3",
            "encoder_id": values["encoder_id"],
            "encoder_revision": None,
            "encoder_dim": values["encoder_dim"],
            "image_size": 224,
            "freeze_encoder": False,
            "trainable_last_layers": 0,
            "gradient_checkpointing": True,
            "spectral_expert": False,
            "use_token_adapter": True,
            "trunk_dim": 512,
        }
    )
    config["training"].update(
        {
            "stage": f"student_dinov3_{variant}",
            "max_updates": values["max_updates"],
            "epochs": values["epochs"],
            "physical_batch_size": 8,
            "gradient_accumulation": 1,
            "required_world_size": 6,
            "optimizer": "adamw",
            "encoder_lr": 2.0e-5,
            "heads_lr": 2.0e-4,
            "layerwise_lr_decay": 0.85,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "generator_balanced_sampler": True,
            "sampler_max_ratio": 5.0,
        }
    )
    config["loss"].update(
        {
            "provenance_original": 1.0,
            "provenance_transformed": 1.0,
            "prediction_consistency": 1.0,
            "feature_consistency": 0.5,
            "mask_focal": 0.0,
            "mask_dice": 0.0,
            "ema_consistency": 2.0,
            "teacher_feature_consistency": 0.5,
        }
    )
    return config


def _slice_provenance_output(
    output: ProvenanceOutput, start: int, end: int | None = None
) -> ProvenanceOutput:
    """Slice every field of a batched structured output without losing metadata."""
    selected = slice(start, end)
    return ProvenanceOutput(
        ai_positive_logit=output.ai_positive_logit[selected],
        probabilities=output.probabilities[selected],
        aigc_features=output.aigc_features[selected],
        tamper_features=output.tamper_features[selected],
        token_tamper_logits=output.token_tamper_logits[selected],
        fusion_gates=None if output.fusion_gates is None else output.fusion_gates[selected],
    )



def train_student(
    teacher_config_path: Path,
    teacher_checkpoint: Path,
    manifest_path: Path,
    output_dir: Path,
    variant: str,
) -> Path:
    project_root = teacher_config_path.resolve().parent.parent
    load_local_environment(project_root)
    teacher_config = load_config(teacher_config_path)
    config = student_config(teacher_config, variant)
    require_mat = bool(config["paths"].get("require_materialized", True))
    verify_materialization(
        manifest_path,
        data_root=config["paths"].get("data_root"),
        allow_missing=not require_mat,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for student distillation")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rank, world_size, local_rank = setup_distributed()
    is_primary = is_main_process()
    required_world_size = int(config["training"].get("required_world_size", 1))
    if world_size != required_world_size:
        cleanup_distributed()
        raise RuntimeError(f"Expected exactly {required_world_size} training processes; got {world_size}")
    device = torch.device(f"cuda:{local_rank}")

    teacher = build_model(teacher_config).to(device).eval()
    _load_checkpoint(teacher, teacher_checkpoint)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = build_model(config).to(device)
    dataset = PairedImageDataset(
        manifest_path,
        data_root=config["paths"].get("data_root"),
        seed=int(config["seed"]),
        transform_families=tuple(
            Transformation[name.upper()]
            for name in config.get("transforms", {}).get("families", [])
        ),
        render_policy=config.get("preprocessing", {}).get("policy", "square_jpeg95"),
    )
    sampler = build_balanced_sampler(
        [int(record.provenance) for record in dataset.records],
        [record.generator for record in dataset.records],
        samples=len(dataset),
        seed=int(config["seed"]),
        rank=rank,
        world_size=world_size,
        ai_positive=[int(record.ai_positive) for record in dataset.records],
        max_ratio=float(config["training"].get("sampler_max_ratio", 5.0)),
    )
    num_workers = int(config["training"].get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["physical_batch_size"]),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(int(config["seed"]) + 10_000),
    )
    if len(loader) == 0:
        cleanup_distributed()
        raise ValueError("Cannot distill a student from an empty manifest")

    optimizer = build_optimizer(student, config)
    student_module: torch.nn.Module = (
        DDP(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
        if world_size > 1
        else student
    )
    training = config["training"]
    loss_config = config["loss"]
    weights = LossWeights(
        provenance_original=float(loss_config.get("provenance_original", 1.0)),
        provenance_transformed=float(loss_config.get("provenance_transformed", 1.0)),
        prediction_consistency=float(loss_config.get("prediction_consistency", 1.0)),
        feature_consistency=float(loss_config.get("feature_consistency", 0.5)),
        mask_focal=float(loss_config.get("mask_focal", 0.0)),
        mask_dice=float(loss_config.get("mask_dice", 0.0)),
        ema_consistency=float(loss_config.get("ema_consistency", 2.0)),
    )
    teacher_feature_weight = float(loss_config.get("teacher_feature_consistency", 0.5))
    teacher_confidence_threshold = float(
        training.get("ema", {}).get("confidence_threshold", 0.8)
    )
    max_updates = int(training["max_updates"])
    accumulation = int(training.get("gradient_accumulation", 1))
    if max_updates <= 0:
        cleanup_distributed()
        raise ValueError("training.max_updates must be positive")
    if accumulation <= 0:
        cleanup_distributed()
        raise ValueError("training.gradient_accumulation must be positive")
    scheduler = build_scheduler(
        optimizer, max_updates, float(training.get("warmup_fraction", 0.05))
    )
    precision = str(training.get("precision", "bf16")).lower()
    if precision not in {"bf16", "float32", "fp32"}:
        cleanup_distributed()
        raise ValueError("training.precision must be bf16 or float32")

    optimizer.zero_grad(set_to_none=True)
    update = 0
    epoch = 0
    last_micro_step = 0
    student_module.train()
    while update < max_updates:
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        micro_steps_per_epoch = len(loader)
        window_loss = 0.0
        window_components: dict[str, float] = {}
        window_micro_steps = 0
        for micro_step, raw_batch in enumerate(loader, start=1):
            if update >= max_updates:
                break
            last_micro_step = micro_step
            is_boundary = (
                micro_step % accumulation == 0 or micro_step == micro_steps_per_epoch
            )
            window_start = ((micro_step - 1) // accumulation) * accumulation + 1
            window_size = min(
                accumulation, micro_steps_per_epoch - window_start + 1
            )
            sync_context = (
                student_module.no_sync()
                if isinstance(student_module, DDP) and not is_boundary
                else nullcontext()
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            )
            with sync_context:
                with autocast:
                    target = raw_batch["ai_positive"].to(device=device, dtype=torch.float32)
                    with torch.no_grad():
                        teacher_output = teacher(raw_batch["original"])
                    original_count = len(raw_batch["original"])
                    combined = student_module(
                        [*raw_batch["original"], *raw_batch["transformed"]]
                    )
                    original = _slice_provenance_output(combined, 0, original_count)
                    transformed = _slice_provenance_output(combined, original_count)
                    loss, components = provenance_robustness_loss(
                        original,
                        transformed,
                        ai_target=target,
                        weights=weights,
                        teacher_probabilities=teacher_output.probabilities.detach(),
                        teacher_confidence_threshold=teacher_confidence_threshold,
                    )
                    teacher_feature_loss = (
                        1.0
                        - F.cosine_similarity(
                            original.provenance_features,
                            teacher_output.provenance_features.detach(),
                            dim=-1,
                        )
                    ).mean()
                    loss = loss + teacher_feature_weight * teacher_feature_loss
                    components = dict(components)
                    components["teacher_feature_consistency"] = teacher_feature_loss
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
                student.parameters(), float(training.get("gradient_clip_norm", 1.0))
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if is_primary and update % 50 == 0:
                divisor = float(window_micro_steps)
                component_text = " ".join(
                    f"{name}={value / divisor:.4f}"
                    for name, value in window_components.items()
                )
                print(
                    f"student={variant} update={update}/{max_updates} "
                    f"loss={window_loss / divisor:.5f} {component_text}",
                    flush=True,
                )
            window_loss = 0.0
            window_components = {}
            window_micro_steps = 0
        epoch += 1

    if world_size > 1:
        torch.distributed.barrier()
    checkpoint = Path()
    if is_primary:
        final_checkpoint = save_checkpoint(
            student,
            config,
            output_dir,
            update,
            optimizer=optimizer,
            scheduler=scheduler,
            manifest_path=manifest_path,
            epoch=epoch - 1,
            micro_step=last_micro_step,
        )
        checkpoint = save_checkpoint(
            student,
            config,
            output_dir,
            update,
            optimizer=optimizer,
            scheduler=scheduler,
            manifest_path=manifest_path,
            epoch=epoch - 1,
            micro_step=last_micro_step,
            filename="checkpoint-best.pt",
            selection={"metric": "teacher_distillation_loss", "update": update},
        )
        if is_primary:
            print(f"student_final_checkpoint={final_checkpoint}", flush=True)
    if world_size > 1:
        torch.distributed.barrier()
    cleanup_distributed()
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a binary DINOv3 ViT-S or ViT-B student from the trained teacher.")
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student", choices=sorted(STUDENT_PRESETS), required=True)
    args = parser.parse_args()
    checkpoint = train_student(
        args.teacher_config,
        args.teacher_checkpoint,
        args.manifest,
        args.output_dir,
        args.student,
    )
    if is_main_process():
        print(f"Student training complete: {checkpoint}")


if __name__ == "__main__":
    main()

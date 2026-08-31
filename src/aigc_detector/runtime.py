from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv


def load_local_environment(project_root: str | Path | None = None) -> None:
    root = Path(project_root) if project_root is not None else Path.cwd()
    # Search project_root, repo root env, and parent directories for .env
    candidates: list[Path] = []
    repo_root = os.environ.get("TECHJAM_REPO_ROOT")
    if repo_root:
        candidates.append(Path(repo_root) / ".env")
    candidates.append(root / ".env")
    for parent in root.parents:
        candidates.append(parent / ".env")
    candidates.append(Path.cwd() / ".env")
    for parent in Path.cwd().parents:
        candidates.append(parent / ".env")
    for candidate in candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    hf_home = os.environ.get("TECHJAM_HF_HOME") or os.environ.get("HF_HOME")
    if hf_home:
        hf_home_path = Path(hf_home).resolve()
        os.environ.setdefault("TECHJAM_HF_HOME", str(hf_home_path))
        os.environ.setdefault("HF_HOME", str(hf_home_path))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home_path / "hub"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home_path / "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home_path / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(project_root) / candidate


def setup_distributed() -> tuple[int, int, int]:
    """Set up multi-GPU training via torchrun / DistributedDataParallel.

    When launched with ``torchrun --nproc_per_node=N``, torch sets RANK,
    WORLD_SIZE, and LOCAL_RANK environment variables.  This function reads
    them and initialises the NCCL process group so all GPUs can synchronise
    gradients during ``backward()``.

    If those variables are absent (plain ``python -m``), we fall back to
    single-GPU mode with rank 0.

    Returns:
        ``(rank, world_size, local_rank)`` — rank is this GPU's global id,
        world_size is the number of GPUs, local_rank is the GPU index on
        this machine (pass it to ``torch.cuda.set_device``).
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    """Shut down the distributed process group (call at end of training)."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    """True on rank 0 — the only process that should print, save, or log."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

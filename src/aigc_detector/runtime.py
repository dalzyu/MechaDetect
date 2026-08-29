from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv


def load_local_environment(project_root: str | Path | None = None) -> None:
    root = Path(project_root) if project_root is not None else Path.cwd()
    load_dotenv(root / ".env", override=False)
    hf_home = os.environ.get("TECHJAM_HF_HOME")
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)


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

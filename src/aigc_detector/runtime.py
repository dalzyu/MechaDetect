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

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_info() -> dict:
    import timm, torchvision
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_sha": git_sha(),
    }
    return info


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config; `overrides` are `key=value` strings (dotted keys ok)."""
    import yaml
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    base = cfg.pop("_base_", None)
    if base:
        parent = load_config(Path(path).parent / base)
        parent.update(cfg)
        cfg = parent
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        cfg[k] = yaml.safe_load(v)
    return cfg

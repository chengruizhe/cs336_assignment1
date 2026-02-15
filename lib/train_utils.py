from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lib.train_config import TrainConfig


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtypes: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"Unsupported dtype `{dtype_name}`. Choices: {list(dtypes)}")
    return dtypes[dtype_name]


def resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    mapping: dict[str, Any] = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
        "int64": np.int64,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported token_dtype `{dtype_name}`. Choices: {list(mapping)}")
    return np.dtype(mapping[dtype_name])


def prepare_experiment_dir(cfg: TrainConfig) -> Path:
    run_id = cfg.checkpoint.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.checkpoint.run_id = run_id
    run_dir = Path(cfg.checkpoint.out_dir) / f"{cfg.checkpoint.run_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_resolved_config(cfg: TrainConfig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))

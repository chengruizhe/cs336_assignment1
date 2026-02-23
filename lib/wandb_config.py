from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypeVar

from lib.config_base import ConfigBase

TConfig = TypeVar("TConfig", bound=ConfigBase)


def load_sweep_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sweep config not found: {p}")

    if p.suffix != ".yaml":
        raise ValueError(
            f"Unsupported sweep config format: {p.suffix}. Only .yaml is supported."
        )
    import yaml  # type: ignore

    data = yaml.safe_load(p.read_text())

    if not isinstance(data, dict):
        raise ValueError("Sweep config must be a mapping at the top level.")
    return data


def apply_wandb_overrides(
    cfg: TConfig,
    overrides: Mapping[str, Any],
) -> TConfig:
    return cfg.with_flat_updates(overrides)

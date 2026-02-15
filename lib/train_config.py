from __future__ import annotations

import argparse
import ast
import json
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int | None = None
    context_length: int = 256
    num_layers: int = 6
    d_model: int = 384
    num_heads: int = 6
    d_ff: int = 1536
    rope_theta: float = 10000.0
    dtype: str = "float32"


@dataclass
class OptimizerConfig:
    # Cos annealing
    min_learning_rate: float = 3e-5
    max_learning_rate: float = 1e-3
    warmup_iters: int = 500
    # AdamW
    alpha: float = 0.001
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    # Gradient clipping
    max_grad_norm: float = 1.0
    zero_grad_set_to_none: bool = True


@dataclass
class DataConfig:
    train_bin_path: str = "data/TinyStoriesV2-GPT4-train-tokens.bin"
    val_bin_path: str = "data/TinyStoriesV2-GPT4-valid-tokens.bin"
    token_dtype: str = "uint16"
    batch_size: int = 16


@dataclass
class LoggingConfig:
    log_interval: int = 10
    eval_interval: int = 200
    eval_batches: int = 20
    use_wandb: bool = True
    wandb_project: str = "cs336-basics"
    wandb_run_name: str = "train-run"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_run_id: str | None = None


@dataclass
class CheckpointConfig:
    out_dir: str = "checkpoints"
    save_interval: int = 500
    run_name: str = "transformer"
    run_id: str | None = None
    resume_from: str | None = None


@dataclass
class TrainConfig:
    seed: int = 42
    max_iters: int | None = None
    total_token_processed: int | None = 32768 * 1000
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self) -> None:
        if self.max_iters is None:
            assert self.total_token_processed is not None
            tokens_per_iter = self.data.batch_size * self.model.context_length
            self.max_iters = (
                self.total_token_processed + tokens_per_iter - 1
            ) // tokens_per_iter
        else:
            assert self.total_token_processed is None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM from scratch.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to .toml or .json config."
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config key(s), e.g. --set model.d_model=512 --set max_iters=1000",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print final config and exit."
    )
    return parser.parse_args()


def load_train_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    merged: dict[str, Any] = {}
    if args.config is not None:
        path = Path(args.config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        merged = _load_config_file(path)
    for override in args.set:
        if "=" not in override:
            raise ValueError(
                f"Invalid --set expression: `{override}` (expected KEY=VALUE)"
            )
        key, raw = override.split("=", 1)
        _deep_set(merged, key, _parse_value(raw))
    _dataclass_update(cfg, merged)
    return cfg


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _deep_set(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = target
    for p in parts[:-1]:
        cursor = cursor.setdefault(p, {})
    cursor[parts[-1]] = value


def _dataclass_update(obj: Any, updates: dict[str, Any]) -> None:
    for f in fields(obj):
        if f.name not in updates:
            continue
        current = getattr(obj, f.name)
        incoming = updates[f.name]
        if is_dataclass(current):
            if not isinstance(incoming, dict):
                raise ValueError(f"Expected dict for nested config field `{f.name}`")
            _dataclass_update(current, incoming)
        else:
            setattr(obj, f.name, incoming)


def _load_config_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text())
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config format: {path.suffix}. Use .toml or .json")

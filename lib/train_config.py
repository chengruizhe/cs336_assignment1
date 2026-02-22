from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar


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
    wandb_run_name: str | None = None
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
    compile_model: bool = False
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    @classmethod
    def load(cls, config_path: str | Path) -> TrainConfig:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.suffix == ".toml":
            data = tomllib.loads(path.read_text())
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
        else:
            raise ValueError(
                f"Unsupported config format: {path.suffix}. Use .toml or .json"
            )

        cfg = cls()
        cfg._update_from_dict(data)
        cfg._finalize()
        return cfg

    def _update_from_dict(self, updates: dict[str, Any]) -> None:
        self._dataclass_update(self, updates)

    def _finalize(self) -> None:
        if not self.logging.wandb_run_name:
            self.logging.wandb_run_name = self.checkpoint.run_name
        if self.max_iters is None:
            if self.total_token_processed is None:
                raise ValueError(
                    "Config must set either `max_iters` or `total_token_processed`."
                )
            tokens_per_iter = self.data.batch_size * self.model.context_length
            self.max_iters = (
                self.total_token_processed + tokens_per_iter - 1
            ) // tokens_per_iter
            return
        if self.total_token_processed is not None:
            raise ValueError(
                "Config must set only one of `max_iters` or `total_token_processed`."
            )

    @classmethod
    def _dataclass_update(cls, obj: Any, updates: dict[str, Any]) -> None:
        for f in fields(obj):
            if f.name not in updates:
                continue
            current = getattr(obj, f.name)
            incoming = updates[f.name]
            if is_dataclass(current):
                if not isinstance(incoming, dict):
                    raise ValueError(
                        f"Expected dict for nested config field `{f.name}`"
                    )
                cls._dataclass_update(current, incoming)
            else:
                setattr(obj, f.name, incoming)

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from lib.config_base import ConfigBase


@dataclass(frozen=True)
class ModelConfig(ConfigBase):
    vocab_size: int | None = None
    context_length: int = 256
    num_layers: int = 6
    d_model: int = 384
    num_heads: int = 6
    d_ff: int = 1536
    rope_theta: float = 10000.0
    dtype: str = "float32"
    mixed_precision: bool = False


@dataclass(frozen=True)
class OptimizerConfig(ConfigBase):
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


@dataclass(frozen=True)
class DataConfig(ConfigBase):
    train_bin_path: str = "data/TinyStoriesV2-GPT4-train-tokens.bin"
    val_bin_path: str = "data/TinyStoriesV2-GPT4-valid-tokens.bin"
    token_dtype: str = "uint16"
    batch_size: int = 16


@dataclass(frozen=True)
class LoggingConfig(ConfigBase):
    log_interval: int = 10
    eval_interval: int = 200
    eval_batches: int = 20
    use_wandb: bool = True
    wandb_project: str = "cs336-basics"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_run_id: str | None = None


@dataclass(frozen=True)
class CheckpointConfig(ConfigBase):
    out_dir: str = "checkpoints"
    save_interval: int = 500
    run_name: str = "transformer"
    run_id: str | None = None
    resume_from: str | None = None


@dataclass(frozen=True)
class TrainConfig(ConfigBase):
    seed: int = 42
    max_iters: int | None = None
    total_token_processed: int | None = None
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

    def with_updates(self, updates: Mapping[str, Any]) -> TrainConfig:
        normalized = dict(updates)
        if "max_iters" not in normalized and (
            self.total_token_processed is not None
            or "total_token_processed" in normalized
        ):
            normalized["max_iters"] = None
        return super().with_updates(normalized)

    def with_flat_updates(self, updates: Mapping[str, Any]) -> TrainConfig:
        normalized = dict(updates)
        if "max_iters" not in normalized and (
            self.total_token_processed is not None
            or "total_token_processed" in normalized
        ):
            normalized["max_iters"] = None
        return super().with_flat_updates(normalized)

    def __post_init__(self) -> None:
        logging_cfg = self.logging
        if not logging_cfg.wandb_run_name:
            logging_cfg = replace(logging_cfg, wandb_run_name=self.checkpoint.run_name)
            object.__setattr__(self, "logging", logging_cfg)

        if self.max_iters is None and self.total_token_processed is None:
            raise ValueError(
                "Config must set either `max_iters` or `total_token_processed`."
            )

        if self.total_token_processed is not None:
            tokens_per_iter = self.data.batch_size * self.model.context_length
            computed_max_iters = (
                self.total_token_processed + tokens_per_iter - 1
            ) // tokens_per_iter
            if self.max_iters is not None and self.max_iters != computed_max_iters:
                raise ValueError(
                    "When `total_token_processed` is set, `max_iters` must match its derived value."
                )
            object.__setattr__(self, "max_iters", computed_max_iters)

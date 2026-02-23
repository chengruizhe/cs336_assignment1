from __future__ import annotations

import argparse
from pathlib import Path

import wandb

from lib.train_config import TrainConfig
from lib.wandb_config import apply_wandb_overrides, load_sweep_config
from scripts.train import Trainer


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return safe.strip("_") or "sweep"


def train_one_run(base_config: str, sweep_folder: str) -> None:
    cfg = TrainConfig.load(base_config)
    sweep_root = Path(cfg.checkpoint.out_dir) / "sweeps" / sweep_folder
    run = wandb.init(
        dir=str(sweep_root),
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        mode=cfg.logging.wandb_mode,
    )
    if run is None:
        raise RuntimeError("wandb.init() did not return a run.")

    sweep_overrides = dict(run.config)
    if "optimizer.max_learning_rate" in sweep_overrides:
        sweep_overrides["optimizer.min_learning_rate"] = (
            0.1 * float(sweep_overrides["optimizer.max_learning_rate"])
        )
    if "logging.wandb_run_name" not in sweep_overrides:
        sweep_overrides["logging.wandb_run_name"] = run.name
    sweep_overrides["checkpoint.out_dir"] = str(sweep_root)

    cfg = apply_wandb_overrides(cfg, sweep_overrides)

    trainer = Trainer(cfg, wandb_run=run)
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run W&B sweeps for training.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base .toml or .json training config.",
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        help="Path to W&B sweep .yaml config. If set, create a new sweep.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Existing sweep ID to run (alternative to --sweep-config).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of runs for this agent process.",
    )
    args = parser.parse_args()

    if bool(args.sweep_config) == bool(args.sweep_id):
        raise ValueError("Provide exactly one of --sweep-config or --sweep-id.")

    sweep_name: str | None = None
    if args.sweep_config:
        sweep_cfg = load_sweep_config(args.sweep_config)
        sweep_id = wandb.sweep(
            sweep=sweep_cfg,
            project=sweep_cfg.get("project"),
            entity=sweep_cfg.get("entity"),
        )
        sweep_name_value = sweep_cfg.get("name")
        if isinstance(sweep_name_value, str) and sweep_name_value:
            sweep_name = sweep_name_value
    else:
        sweep_id = args.sweep_id

    sweep_suffix = sweep_id.split("/")[-1]
    if sweep_name is not None:
        sweep_folder = f"{_sanitize_name(sweep_name)}_{sweep_suffix}"
    else:
        sweep_folder = sweep_suffix

    wandb.agent(
        sweep_id,
        function=lambda: train_one_run(args.config, sweep_folder),
        count=args.count,
    )


if __name__ == "__main__":
    main()

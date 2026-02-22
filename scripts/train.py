from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import CrossEntropyLoss
from cs336_basics.transformer import Transformer
from lib.memmap_dataset import MemmapTokenDataset
from lib.train_config import TrainConfig
from lib.train_utils import (
    prepare_experiment_dir,
    resolve_device,
    resolve_numpy_dtype,
    resolve_torch_dtype,
    save_resolved_config,
)
from tests.adapters import (
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
)


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_torch_dtype(cfg.model.dtype)
        tqdm.write(
            "Runtime: "
            f"device={self.device}, dtype={self.dtype}, "
            f"mps_available={torch.backends.mps.is_available()}"
        )
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.run_dir = prepare_experiment_dir(self.cfg)
        self.train_data, self.val_data = self._load_datasets()
        assert (
            self.cfg.model.vocab_size is not None
        ), "model.vocab_size must be specified in config."
        vocab_size = self.cfg.model.vocab_size
        self.model = Transformer(
            vocab_size=vocab_size,
            context_length=cfg.model.context_length,
            num_layers=cfg.model.num_layers,
            d_model=cfg.model.d_model,
            num_heads=cfg.model.num_heads,
            d_ff=cfg.model.d_ff,
            rope_theta=cfg.model.rope_theta,
            device=self.device,
            dtype=self.dtype,
        ).to(self.device)
        self.checkpoint_model = self.model
        self.model = self._compile_model(self.model)
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = AdamW(
            self.checkpoint_model.parameters(),
            lr=cfg.optimizer.alpha,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps,
        )
        self.global_step = 0
        self.wandb_run = self._init_wandb()
        save_resolved_config(self.cfg, self.run_dir / "config.resolved.json")

    def train(self) -> None:
        if self.cfg.checkpoint.resume_from:
            resume_path = Path(self.cfg.checkpoint.resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
            self.global_step = run_load_checkpoint(
                resume_path, self.checkpoint_model, self.optimizer
            )
            tqdm.write(f"Resumed from {resume_path} at iteration {self.global_step}")

        t0 = time.time()
        progress = tqdm(
            total=self.cfg.max_iters,
            initial=self.global_step,
            dynamic_ncols=True,
            desc="train",
        )
        try:
            while self.global_step < self.cfg.max_iters:
                self.global_step += 1
                lr = self._lr_for_step(self.global_step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                self.model.train()
                x, y = self.train_data.sample_batch(
                    batch_size=self.cfg.data.batch_size,
                    context_length=self.cfg.model.context_length,
                    device=self.device,
                )

                logits = self.model(x)
                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                self.optimizer.zero_grad(
                    set_to_none=self.cfg.optimizer.zero_grad_set_to_none
                )
                loss.backward()
                run_gradient_clipping(
                    self.checkpoint_model.parameters(), self.cfg.optimizer.max_grad_norm
                )
                self.optimizer.step()
                progress.update(1)

                train_loss = float(loss.detach().cpu().item())
                if (
                    self.global_step % self.cfg.logging.log_interval == 0
                    or self.global_step == 1
                ):
                    elapsed = time.time() - t0
                    it_s = self.global_step / max(elapsed, 1e-8)
                    progress.set_postfix(
                        {
                            "loss": f"{train_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "it/s": f"{it_s:.2f}",
                        }
                    )
                    self._log(
                        {
                            "train/loss": train_loss,
                            "train/lr": lr,
                            "perf/iter_per_sec": it_s,
                        },
                        step=self.global_step,
                    )

                if (
                    self.global_step % self.cfg.logging.eval_interval == 0
                    or self.global_step == self.cfg.max_iters
                ):
                    val_loss = self.evaluate(self.cfg.logging.eval_batches)
                    progress.set_postfix(
                        {
                            "loss": f"{train_loss:.4f}",
                            "val": f"{val_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        }
                    )
                    self._log({"val/loss": val_loss}, step=self.global_step)

                if (
                    self.global_step % self.cfg.checkpoint.save_interval == 0
                    or self.global_step == self.cfg.max_iters
                ):
                    ckpt_path = self._checkpoint_path(self.global_step)
                    run_save_checkpoint(
                        self.checkpoint_model,
                        self.optimizer,
                        self.global_step,
                        ckpt_path,
                    )
                    tqdm.write(f"Saved checkpoint: {ckpt_path}")
        finally:
            progress.close()

        if self.wandb_run is not None:
            self.wandb_run.finish()

    @torch.no_grad()
    def evaluate(self, num_batches: int) -> float:
        self.model.eval()
        losses: list[float] = []
        for _ in range(num_batches):
            x, y = self.val_data.sample_batch(
                batch_size=self.cfg.data.batch_size,
                context_length=self.cfg.model.context_length,
                device=self.device,
            )
            logits = self.model(x)
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            losses.append(float(loss.detach().cpu().item()))
        return sum(losses) / len(losses)

    def _load_datasets(self) -> tuple[MemmapTokenDataset, MemmapTokenDataset]:
        train_path = Path(self.cfg.data.train_bin_path)
        if not train_path.exists():
            raise FileNotFoundError(f"Train bin not found: {train_path}")
        val_path = Path(self.cfg.data.val_bin_path)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation bin not found: {val_path}")
        np_dtype = resolve_numpy_dtype(self.cfg.data.token_dtype)
        train_arr = np.memmap(train_path, mode="r", dtype=np_dtype)
        val_arr = np.memmap(val_path, mode="r", dtype=np_dtype)

        min_tokens = self.cfg.model.context_length + 2
        if len(train_arr) < min_tokens:
            raise ValueError(
                f"Train tokens ({len(train_arr)}) are too small for context_length={self.cfg.model.context_length}"
            )
        if len(val_arr) < min_tokens:
            raise ValueError(
                f"Validation tokens ({len(val_arr)}) are too small for context_length={self.cfg.model.context_length}"
            )

        train_ds = MemmapTokenDataset(train_arr, 0, len(train_arr))
        val_ds = MemmapTokenDataset(val_arr, 0, len(val_arr))
        tqdm.write(
            f"Loaded train tokens={len(train_ds):,} val tokens={len(val_ds):,} dtype={np_dtype}"
        )
        return train_ds, val_ds

    def _checkpoint_path(self, step: int) -> Path:
        return self.run_dir / f"checkpoint_iter{step:07d}.pt"

    def _lr_for_step(self, it: int) -> float:
        return float(
            run_get_lr_cosine_schedule(
                it=it,
                max_learning_rate=self.cfg.optimizer.max_learning_rate,
                min_learning_rate=self.cfg.optimizer.min_learning_rate,
                warmup_iters=self.cfg.optimizer.warmup_iters,
                cosine_cycle_iters=self.cfg.max_iters,
            )
        )

    def _init_wandb(self):
        if not self.cfg.logging.use_wandb:
            return None

        import wandb  # type: ignore

        run = wandb.init(
            dir=self.run_dir,
            project=self.cfg.logging.wandb_project,
            entity=self.cfg.logging.wandb_entity,
            name=self.cfg.logging.wandb_run_name,
            mode=self.cfg.logging.wandb_mode,
            config=asdict(self.cfg),
        )
        self.cfg.logging.wandb_run_id = run.id
        save_resolved_config(self.cfg, self.run_dir / "config.resolved.json")
        return run

    def _compile_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.cfg.compile_model:
            return model
        compile_backend = (
            "aot_eager" if self.device == torch.device("mps") else "inductor"
        )
        compiled = torch.compile(
            model,
            mode=self.cfg.compile_mode,
            fullgraph=self.cfg.compile_fullgraph,
            dynamic=self.cfg.compile_dynamic,
            backend=compile_backend,
        )
        tqdm.write(
            "Enabled torch.compile "
            f"(mode={self.cfg.compile_mode}, "
            f"fullgraph={self.cfg.compile_fullgraph},"
            f"dynamic={self.cfg.compile_dynamic},"
            f"backend={compile_backend})"
        )
        return compiled

    def _log(self, metrics: dict[str, float], step: int) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer LM from config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to .toml or .json training config.",
    )
    args = parser.parse_args()

    cfg = TrainConfig.load(args.config)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()

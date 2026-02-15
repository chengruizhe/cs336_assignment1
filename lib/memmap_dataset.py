from __future__ import annotations

import numpy as np
import torch

from tests.adapters import run_get_batch


class MemmapTokenDataset:
    def __init__(self, memmap: np.memmap, start: int = 0, end: int | None = None) -> None:
        end_idx = len(memmap) if end is None else end
        if end_idx - start <= 1:
            raise ValueError(f"Dataset slice is too small: start={start}, end={end_idx}")
        self._arr = memmap[start:end_idx]

    def __len__(self) -> int:
        return len(self._arr)

    def sample_batch(
        self,
        *,
        batch_size: int,
        context_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = run_get_batch(
            dataset=self._arr,
            batch_size=batch_size,
            context_length=context_length,
            device="cpu",
        )
        if device.type != "cpu":
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        return x, y

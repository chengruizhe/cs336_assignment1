from collections.abc import Callable
from typing import Any
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        eps: float = 1e-8,
    ) -> None:
        assert lr > 0
        assert betas[0] > 0 and betas[1] > 0, betas
        assert weight_decay > 0
        assert eps > 0
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> Any:
        loss = None if closure is None else closure()
        t = getattr(self, "t", 1)
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = getattr(self, "t", 1)
                m = state.get("m", 0.0)
                v = state.get("v", 0.0)

                grad = p.grad.data

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                alpha = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.data -= alpha * m / (v**0.5 + eps) + lr * weight_decay * p.data

                state["m"] = m
                state["v"] = v
        
        self.t = t + 1
        return loss

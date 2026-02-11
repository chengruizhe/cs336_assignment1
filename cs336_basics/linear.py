import torch
import torch.nn as nn
import einx


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=True,
        )
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return einx.dot("... i, o i -> ... o", input, self.weight)

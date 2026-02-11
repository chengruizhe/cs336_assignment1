import torch
import torch.nn as nn
import einx


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones((d_model), device=device, dtype=dtype),
            requires_grad=True,
        )
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(einx.mean("... [d] -> ...", x**2) + self.eps)
        result = einx.multiply("... d, ..., d -> ... d", x, 1 / rms, self.weight)
        return result.to(dtype=input_dtype)

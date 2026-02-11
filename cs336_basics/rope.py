import torch
import torch.nn as nn
import einx
from torch import Tensor
from jaxtyping import Float, Int


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        k = torch.arange(0, d_k // 2, device=device)
        inv_freq = 1.0 / (theta ** (2 * k / d_k))

        positions = torch.arange(0, max_seq_len, device=device)
        freqs = torch.outer(positions, inv_freq)  # shape: (max_seq_len, d_k // 2)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... s d_k"],
        token_positions: Int[Tensor, "... s"],
    ) -> Float[Tensor, "... s d_k"]:
        cos = einx.get_at(
            "[max_s] d_half, ... s -> ... s d_half",
            self.cos,
            token_positions,
        )
        sin = einx.get_at(
            "[max_s] d_half, ... s -> ... s d_half",
            self.sin,
            token_positions,
        )

        x_even, x_odd = einx.rearrange(
            "... s (d_half two) -> two ... s d_half",
            x,
            two=2,
        )

        r_even = x_even * cos - x_odd * sin
        r_odd = x_even * sin + x_odd * cos
        result = torch.stack((r_even, r_odd), dim=-1)
        return einx.rearrange(
            "... s d_half two -> ... s (d_half two)",
            result,
        )

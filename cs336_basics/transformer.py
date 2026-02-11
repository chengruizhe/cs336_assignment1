import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from .rope import RotaryPositionalEmbedding
from .rmsnorm import RMSNorm
from .attention import MultiHeadSelfAttention
from .swiglu import SwiGLU
from .embedding import Embedding
from .linear import Linear


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            in_features=d_model,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... s d_model"],
    ) -> Float[Tensor, ".. s d_model"]:
        token_positions = torch.arange(x.shape[-2], device=x.device)
        layer1 = x + self.attn(self.ln1(x), token_positions=token_positions)
        return layer1 + self.ffn(self.ln2(layer1))


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope=rope,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_indices: Int[Tensor, "b s"],
    ) -> Float[Tensor, "... s d_model"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

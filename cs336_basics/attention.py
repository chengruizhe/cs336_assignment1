import torch
from torch import Tensor
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
import einx

from .softmax import Softmax
from .rope import RotaryPositionalEmbedding
from .linear import Linear


class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = Softmax()

    def forward(
        self,
        Q: Float[Tensor, "... queries d_k"],
        K: Float[Tensor, "... keys d_k"],
        V: Float[Tensor, "... values d_v"],
        mask: Bool[Tensor, "... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
        QK = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", Q, K)
        d_k = Q.shape[-1]
        assert K.shape[-2] == V.shape[-2]

        scores = QK / (d_k**0.5)
        neg_inf = torch.finfo(scores.dtype).min
        if mask is not None:
            scores = scores.masked_fill(~mask, neg_inf)
        return einx.dot(
            "... queries keys, ... keys d_v -> ... queries d_v",
            self.softmax(scores, dim=-1),
            V,
        )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        in_features: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert self.d_k * num_heads == d_model, "d_model must be divisible by num_heads"

        self.qkv_proj = Linear(
            in_features=in_features,
            out_features=d_model * 3,
            device=device,
            dtype=dtype,
        )

        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.rope = rope
        self.sdpa = ScaledDotProductAttention()

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_in"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_out"]:
        seq_len = x.shape[-2]
        x = self.qkv_proj(x)
        Q, K, V = einx.rearrange(
            "... s (three h d_k) -> three ... h s d_k",
            x,
            h=self.num_heads,
            three=3,
            d_k=self.d_k,
        )

        if self.rope is not None:
            assert token_positions is not None
            assert token_positions.shape[-1] == seq_len, token_positions.shape
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(
            torch.ones(
                (seq_len, seq_len),
                dtype=torch.bool,
                device=x.device,
            )
        )
        output = self.sdpa(Q=Q, K=K, V=V, mask=mask)
        output = einx.rearrange(
            "... h s d_v -> ... s (h d_v)",
            output,
        )
        return self.output_proj(output)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        q_key = prefix + "q_proj.weight"
        k_key = prefix + "k_proj.weight"
        v_key = prefix + "v_proj.weight"
        qkv_key = prefix + "qkv_proj.weight"

        if q_key in state_dict and k_key in state_dict and v_key in state_dict:
            # Merge into single QKV weight
            state_dict[qkv_key] = torch.cat(
                [state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0
            )
            del state_dict[q_key]
            del state_dict[k_key]
            del state_dict[v_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

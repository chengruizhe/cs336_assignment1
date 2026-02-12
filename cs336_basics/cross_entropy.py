import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
import einx

class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(
        self,
        x: Float[Tensor, "b v"],
        targets: Int[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        x = x - torch.max(x, dim=-1, keepdim=True).values
        logits = einx.get_at(
            "b [v], b -> b",
            x,
            targets,
        )
        x_sum = torch.sum(torch.exp(x), dim=-1)
        return torch.mean(torch.log(x_sum) - logits)
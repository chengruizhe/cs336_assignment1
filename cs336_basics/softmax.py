import torch
import torch.nn as nn


class Softmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        x = x - torch.max(x, dim=dim, keepdim=True).values
        x_sum = torch.sum(torch.exp(x), dim=dim, keepdim=True)
        return torch.exp(x) / x_sum

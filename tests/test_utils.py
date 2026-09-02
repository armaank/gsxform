"""test utility functions"""

import torch


def create_adj(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Generate a random symmetric adjacency matrix."""
    W = torch.triu((x > p).float(), diagonal=1)

    return W + W.transpose(-2, -1)


def sqrt_degree(W_adj: torch.Tensor) -> torch.Tensor:
    """Normalized square-root degree vector."""
    v = torch.sqrt(W_adj.sum(1))

    return v / v.norm(dim=-1, keepdim=True)

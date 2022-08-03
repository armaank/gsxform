"""test utility functions
"""

import torch


def create_adj(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Utility function used to create random a adjacency matrix."""
    return (x > p).float()

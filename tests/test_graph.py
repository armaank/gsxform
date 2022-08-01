"""testing suite for graph.py
"""
import torch

from gsxform import graph


def create_adj(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Utility function used to create random a adjacency matrix."""
    return (x > p).float()


def test_adjacency_to_laplacian():  # type: ignore
    """Test graph.adjacency_to_laplacian."""

    x = torch.rand((16, 10, 10))

    W_adj = create_adj(x)

    L = graph.adjacency_to_laplacian(W_adj)

    assert L.size() == W_adj.size()


def test_normalize_adjacency():  # type: ignore
    """Test graph.normalize_adjacency for shape and numerical stability."""

    x = torch.rand((16, 10, 10))

    W_adj = create_adj(x)

    W_norm = graph.normalize_adjacency(W_adj)

    assert W_adj.size() == W_norm.size()

    is_nan = torch.isnan(W_norm)

    assert not torch.any(is_nan)


def test_normalize_laplacian():  # type: ignore
    """Test graph.normalize_adjacency for shape and numerical stability."""

    x = torch.rand((16, 10, 10))

    W_adj = create_adj(x)

    L = graph.adjacency_to_laplacian(W_adj)

    L_norm = graph.normalize_laplacian(L)

    assert L.size() == L_norm.size()

    is_nan = torch.isnan(L_norm)

    assert not torch.any(is_nan)

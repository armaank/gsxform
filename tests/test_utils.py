"""test utility functions"""

import torch


def create_adj(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Generate a random symmetric adjacency matrix."""
    W = torch.triu((x > p).float(), diagonal=1)

    return W + W.transpose(-2, -1)


def pad_graphs(
    sizes: list[int], n_features: int = 3
) -> tuple[
    list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build graphs of the given sizes, and their zero-padded batch and mask."""
    graphs = [create_adj(torch.rand((1, n, n)))[0] for n in sizes]
    signals = [torch.rand((n_features, n)) for n in sizes]

    n_max = max(sizes)
    W_adj = torch.zeros((len(sizes), n_max, n_max))
    x = torch.zeros((len(sizes), n_features, n_max))
    mask = torch.zeros((len(sizes), n_max), dtype=torch.bool)
    for ii, n in enumerate(sizes):
        W_adj[ii, :n, :n] = graphs[ii]
        x[ii, :, :n] = signals[ii]
        mask[ii, :n] = True

    return graphs, signals, W_adj, x, mask


def sqrt_degree(W_adj: torch.Tensor) -> torch.Tensor:
    """Normalized square-root degree vector."""
    v = torch.sqrt(W_adj.sum(1))

    return v / v.norm(dim=-1, keepdim=True)

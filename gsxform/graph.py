"""Graph utility functions."""

import torch
from einops import einsum


def adjacency_to_laplacian(W: torch.Tensor) -> torch.Tensor:
    """Convert an adjacency matrix into the graph Laplacian.

    Parameters
    ----------
    W: torch.Tensor
        Batch of normalized graph adjacency matricies.

    Returns
    -------
    torch.Tensor
        Batch of graph Laplacians.

    """
    L = torch.diag_embed(W.sum(1)) - W

    return L


def normalize_adjacency(W: torch.Tensor) -> torch.Tensor:
    """Normalize an adjacency matrix.

    Parameters
    ----------
    W: torch.Tensor
        Batch of adjacency matricies.

    Returns
    -------
    torch.Tensor
        Batch of normalized adjacency matricies.

    """
    # build degree vector
    d = W.sum(1)
    # normalize
    d_invsqrt = 1.0 / torch.sqrt(torch.max(torch.ones(d.size(), device=d.device), d))
    W_norm: torch.Tensor = einsum(d_invsqrt, W, d_invsqrt, "b i, b i j, b j -> b i j")

    return W_norm


def lazy_diffusion(W: torch.Tensor) -> torch.Tensor:
    """Build the lazy diffusion operator T = 1/2 (I+A).

    Parameters
    ----------
    W: torch.Tensor
        Batch of adjacency matricies.

    Returns
    -------
    torch.Tensor
        Batch of lazy diffusion operators.

    """
    I_N = torch.eye(W.shape[-1], device=W.device)

    T = 1 / 2 * (I_N + normalize_adjacency(W))

    return T


def normalize_laplacian(L: torch.Tensor) -> torch.Tensor:
    """Normalize an graph Laplacian.

    Parameters
    ----------
    L: torch.Tensor
        Batch of graph Laplacians.

    Returns
    -------
    torch.Tensor
        Batch of normalized graph Laplacians.
    """
    # build degree vector
    # batch diagonal
    # (https://pytorch.org/docs/stable/generated/torch.diagonal.html#torch.diagonal)
    d = torch.diagonal(L, dim1=-2, dim2=-1)
    # normalize
    d_invsqrt = 1.0 / torch.sqrt(torch.max(torch.ones(d.size(), device=d.device), d))
    L_norm: torch.Tensor = einsum(d_invsqrt, L, d_invsqrt, "b i, b i j, b j -> b i j")

    return L_norm


def compute_spectra(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the spectra of graph Laplacian from its adjacency matrix.

    Performs an eigendecomposition (w/o assuming additional structure)
    using `torch.linalg.eigh` (previously used `torch.symeig`) on a normalized
    graph laplacian. Converts from the adjacency matrix to the laplacian
    internally.

    Parameters
    ----------
    W: torch.Tensor
        Batch of graph adjacency matricies.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Batch of eigenvalues and eigenvectors of the graph Laplacian.
    """
    # compute laplacian
    L = adjacency_to_laplacian(W)
    # normalize laplacian
    L_norm = normalize_laplacian(L)
    # perform eigen decomp
    # come out in ascending order,
    E, V = torch.linalg.eigh(L_norm, UPLO="L")

    return E, V

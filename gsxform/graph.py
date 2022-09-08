"""graph utility functions.
"""

from typing import Tuple

import torch


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
    D_invsqrt = torch.diag_embed(1.0 / torch.sqrt(torch.max(torch.ones(d.size()), d)))
    W_norm = D_invsqrt.matmul(W).matmul(D_invsqrt)

    return W_norm


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
    D_invsqrt = torch.diag_embed(1.0 / torch.sqrt(torch.max(torch.ones(d.size()), d)))
    L_norm = D_invsqrt.matmul(L).matmul(D_invsqrt)

    return L_norm


def compute_spectra(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

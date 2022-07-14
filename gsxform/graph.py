"""graph utility functions

TODO:

"""

from typing import Tuple

import torch


def adjacency_to_laplacian(W: torch.Tensor) -> torch.Tensor:
    """convert adjacency matrix into the graph Laplacian

    Parameters
    ----------
    W: torch.Tensor
        Batch of normalized graph adjacency matricies

    Returns
    -------
    torch.Tensor
        Batch of graph Laplacians

    """

    L = torch.diag_embed(W.sum(1)) - W

    return L


def normalize_adjacency(W: torch.Tensor) -> torch.Tensor:
    """normalize adjacency matrix

    Parameters
    ----------
    W: torch.Tensor
        Batch of adjacency matricies

    Returns
    -------
    torch.Tensor
        Batch of normalized adjacency matricies

    """

    # build degree vector (diag(L) is the same as W.sum(1)
    d = W.sum(1)
    # normalize
    D_invsqrt = torch.diag_embed(1.0 / torch.sqrt(torch.max(torch.ones(d.size()), d)))
    W_norm = D_invsqrt.matmul(W).matmul(D_invsqrt)

    return W_norm


def normalize_laplacian(L: torch.Tensor) -> torch.Tensor:
    """normalize graph Laplacian

    Parameters
    ----------
    L: torch.Tensor
        Batch of graph Laplacians

    Returns
    -------
    torch.Tensor
        Batch of normalized graph Laplacians
    """

    # build degree vector (diag(L) is the same as W.sum(1)
    d = torch.diag(L)
    # normalize
    D_invsqrt = torch.diag_embed(1.0 / torch.sqrt(torch.max(torch.ones(d.size()), d)))
    L_norm = D_invsqrt.matmul(L).matmul(D_invsqrt)

    return L_norm


def compute_spectra(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """computes spectra of graph laplacian from its adjacency matrix

    Parameters
    ----------
    W: torch.Tensor
       Batch of graph adjacency matricies

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Batch of eigenvalues and eigenvectors of the graph laplacian
    """

    # compute laplacian
    L = adjacency_to_laplacian(W)
    # normalize laplacian
    L_norm = normalize_laplacian(L)
    # perform eigen decomp
    E, V = torch.symeig(L_norm, eigenvectors=True)

    return E, V

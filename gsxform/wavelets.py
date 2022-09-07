"""Implementations of graph wavelet transforms and kernel functions

TODO:
    - figure out wavelet vs kernel notation
    - consider making wavelet class
        - get_kernel, matmul, append pattern
        - could add checks
    - add references
    - single variable names
"""

import torch
from einops import rearrange

from .graph import compute_spectra
from .kernel import TightHannKernel


def diffusion_wavelets(T: torch.Tensor, n_scales: int) -> torch.Tensor:
    """Compute diffusion wavelet filter bank

    Computes diffusion wavelets from inputs. ADD DOCS, references

    Parameters
    ----------
    T: torch.Tensor
        Input diffusion matrix computed from adjacency matrix
    n_scales: int
        Number of scales to use in wavelet transform

    Returns
    -------
    phi: torch.Tensor
        wavelet filter bank
    """

    # make n_node x n_node identity matrix
    I_N = torch.eye(T.shape[1])

    # compute zero-eth order (J=0) wavelet filter
    # one half the normalized laplacian operator 1/2(I-D^-1/2WD^-1/2)
    psi = I_N - T

    for jj in range(1, n_scales):
        # compute jth diffusion operator (wavelet kernel)
        T_j = torch.matrix_power(T, 2 ** (jj - 1))
        # compute jth wavelet filter via matmul
        psi_j = torch.einsum("b n m, b n m -> b n m", T_j, (I_N - T_j))
        # append wavelets
        psi = torch.cat((psi, psi_j), axis=0)

    psi = rearrange(psi, "(b ns) ni nj -> b ns ni nj", ns=n_scales)

    return psi


def tighthann_wavelets(
    W_adj: torch.Tensor, n_scales: int, kernel: TightHannKernel
) -> torch.Tensor:
    """Computes spectrum adapted tight Hann wavelets.

    Parameters
    ----------
    W_adj: torch.Tensor
        Input batch of adjacency matricies
    n_scales: int
        Number of scales to use in wavelet transform
    kernel: TightHannKernel
        Adaptive kernel used in wavelet transform.

    Returns
    -------
    psi: torch.Tensor
        wavelet filter bank

    """
    E, V = compute_spectra(W_adj)

    V_herm = rearrange(V, "b ni nj -> b nj ni")  # hermetian transpose

    # compute wavelet coeffs
    # check loop bounds, iterators...
    psi = torch.zeros_like(V)

    for jj in range(1, n_scales):

        # compute kernel, check notation here...
        phi = torch.diag_embed(kernel.adapted_kernels(E, jj))
        # compute jth wavelet filter via matmul
        psi_j = torch.einsum("b n m, b n m, b n m -> b n m", V, phi, V_herm)
        # append wavelets
        psi = torch.cat((psi, psi_j), axis=0)

    psi = rearrange(psi, "(b ns) ni nj -> b ns ni nj", ns=n_scales)
    return psi

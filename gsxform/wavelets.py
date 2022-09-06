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


def diffusion_wavelets(T: torch.Tensor, n_scales: int) -> torch.Tensor:
    """Compute diffusion wavelet filter bank

    Computes diffusion wavelets from inputs. ADD DOCS, references

    Parameters
    ----------
    T: torch.Tensor
        Input diffusion matrix computed from adjacency matrix
    n_scales: int
        Number of scales to use in wavelet transform
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

"""Implementations of graph wavelet transforms and kernel functions
"""
import torch


def diffusion_wavelets(T: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """compute filters for diffusion wavelets

    Parameters
    ----------
    T: torch.Tensor
        diffusion operator
    J: torch.Tensor
        number of scales (integer)


    Returns
    -------
    torch.Tensor
        wavelet transforms for each scale
    """

    # number of nodes
    N = T.shape[0]
    I_N = torch.eye(N)

    # compute zero-eth order (J=0) wavelet filter
    # one half the normalized laplacian operator 1/2(I-D^-1/2WD^-1/2)
    psi_0 = I_N - T

    # reshape for loop
    psi = psi_0.reshape(1, N, N)

    for jj in range(1, J):
        # compute jth diffusion operator (wavelet kernel)
        T_j = torch.matrix_power(T, 2 ** (jj - 1))
        # compute jth wavelet filter
        psi_j = torch.matmul(T_j, (I_N - T_j))
        # append wavelets
        psi = torch.cat(psi, psi_j.reshape(1, N, N), axis=0)

    return psi


# def spline_wavelets(J: torch.Tensor,

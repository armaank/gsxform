"""Implementations of graph wavelet transforms and kernel functions

TODO:
    - figure out wavelet vs kernel notation
    - consider making wavelet class
        - get_kernel, matmul, append pattern
        - could add checks
    - add references
"""
import numpy as np
import torch

from .kernel import hann_kernel, spline_kernel


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
    N = T.shape[1]  # batch in first layer, might need to change this?
    I_N = torch.eye(N)

    # compute zero-eth order (J=0) wavelet filter
    # one half the normalized laplacian operator 1/2(I-D^-1/2WD^-1/2)
    psi_0 = I_N - T

    # reshape for loop
    # changed to batch size, ....
    b = T.shape[0]
    psi = psi_0.reshape(b, N, N)

    for jj in range(1, J):
        # compute jth diffusion operator (wavelet kernel)
        T_j = torch.matrix_power(T, 2 ** (jj - 1))
        # compute jth wavelet filter
        psi_j = torch.matmul(T_j, (I_N - T_j))
        # append wavelets
        psi = torch.cat((psi, psi_j.reshape(b, N, N)), axis=0)

    return psi


def spline_wavelets(
    V: torch.Tensor,
    E: torch.Tensor,
    J: int,
    alpha: int,
    beta: int,
    x1: int,
    x2: int,
    K: int,
    gamma: float,
) -> torch.Tensor:
    """compute cubic spline wavelets

    TODO: double check types for kernel params

    Parameters
    ----------

    V: torch.tensor
        matrix of eigenvectors of the graph Laplacian
    E: torch.Tensor
        diagonal matrix of eigenvalues of the graph Laplacian
    J: int
        nuumber of scales
    alpha: int
        spline kernel parameter
    beta: int
        spline kernel parameter
    x1: int
        spline kernel parameter
    x2: ing
        spline kernel parameter
    K: int
        design parameter used to scale the maximum eigenvalue
    gamma: float:
        maximum eigenvalue

    Returns
    -------
    torch.Tensor
        wavelet transforms for each scale
    """

    # get eigenvalues from square matrix
    eigs = torch.diagonal(E)  # check this
    # compute hermitian transpose of eigenvectors
    V_adj = V.adjoint()

    eig_max = gamma
    eig_min = gamma / K

    # scales based on eigenvalue spread
    t = torch.logspace(np.log10(x2 / eig_min), np.log10(x2 / eig_max), J - 1)

    # init wavelet matrix
    N = V.shape[2]  # realized that this was off?

    # compute zero-eth order filter
    psi_0 = torch.exp((-(eigs / 0.6 * eig_min)) ** 4)
    psi_0 = psi_0.squeeze()

    # reshape for loop
    psi = torch.empty([0, N, N])

    # compute wavelet filter bank
    for jj in range(0, J - 1):  # check loop bounds

        psi_j = spline_kernel(t[jj] * eigs, alpha, beta, x1, x2).to(torch.float)
        psi_j = torch.matmul(torch.matmul(V, torch.diag_embed(psi_j)), V_adj).reshape(
            1, N, N
        )

        psi = torch.cat((psi, psi_j), axis=0)
    # compute zero-eth order
    psi_0 = torch.matmul(
        torch.matmul(V, torch.diag_embed(torch.max(torch.abs(psi)) * psi_0)), V_adj
    ).reshape(1, N, N)
    psi = torch.concat((psi_0, psi), axis=0)

    return psi


def hann_wavelets(
    V: torch.Tensor,
    E: torch.Tensor,
    J: int,
    R: int,
    gamma: float,
    warp: bool,
) -> torch.Tensor:
    """compute tight hann wavelets

    Parameters
    ----------

    V: torch.tensor
        matrix of eigenvectors of the graph Laplacian
    E: torch.Tensor
        diagonal matrix of eigenvalues of the graph Laplacian
    J: int
        nuumber of scales
    R: int
        scale factor used in eq. 9 of Shuman et. al
    gamma: float:
        maximum eigenvalue
    warp: bool:
        optional adapt wavelets to spectrum

    Returns
    -------
    torch.Tensor
        wavelet transforms for each scale

    """
    eigs = E
    # compute hermentian transpose of eigenvectors
    V_adj = V.adjoint()
    # init wavelet matrix
    N = V.shape[2]
    b = V.shape[0]

    eig_max = gamma  # might want to compute this here?
    if warp:
        eigs = torch.log(eigs)  # check numerical stability
        eig_max = torch.log(eig_max)
        power = torch.zeros((b, N))  # scaled by power spectra
    # scales based on uniform translates
    t = torch.arange(1, J + 1) * eig_max / (J + 1 - R)

    assert V.shape[1] == V.shape[2]

    psi = torch.empty([0, N, N])

    # compute wavelet filter bank
    for jj in range(0, J - 1):

        # no warping, K is fixed to 1
        psi_j = hann_kernel(eigs - t[jj], J, R, eig_max)
        if warp:
            power += torch.abs(psi_j) ** 2

        psi_j = torch.matmul(torch.matmul(V, torch.diag_embed(psi_j)), V_adj).reshape(
            b, N, N
        )
        psi = torch.cat((psi, psi_j), axis=0)
    if warp:
        psi_J = R * 0.25 + R / 2 * 0.25 - power
        psi_J = torch.sqrt(psi_J)
        psi_J = torch.matmul(torch.matmul(V, torch.diag_embed(psi_J)), V_adj).reshape(
            b, N, N
        )
        psi = torch.cat((psi_J, psi), axis=0)
    # computing final filter, need to double check this
    else:
        psi_J = hann_kernel(eigs - t[J - 1], J, R, eig_max)  # check this indexing
        psi_J = torch.matmul(torch.matmul(V, torch.diag_embed(psi_J)), V_adj).reshape(
            b, N, N
        )
        psi = torch.cat((psi, psi_J), axis=0)

    return psi

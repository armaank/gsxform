"""Implementations of graph wavelet transforms and kernel functions

TODO:
    - figure out wavelet vs kernel notation
    - consider making wavelet class
        - get_kernel, matmul, append pattern
        - could add checks
    - add references
"""
import torch


def diffusion_wavelets(W: torch.Tensor, n_scales: int) -> torch.Tensor:

    n_nodes = W.shape[1]

    diag = W.sum(1)
    dhalf = torch.diag_embed(1.0 / torch.sqrt(torch.max(torch.ones(diag.size()), diag)))
    # L = torch.diag_embed(W.sum(1)) - W
    W_norm = dhalf.matmul(W).matmul(dhalf)
    print(W_norm.shape)
    print(torch.eye(n_nodes).shape)
    T = 1 / 2 * (torch.eye(n_nodes) + W_norm)

    # number of nodes
    N = T.shape[1]  # batch in first layer, might need to change this?
    print(N)
    I_N = torch.eye(N)

    # compute zero-eth order (J=0) wavelet filter
    # one half the normalized laplacian operator 1/2(I-D^-1/2WD^-1/2)
    psi_0 = I_N - T

    # reshape for loop
    # changed to batch size, ....
    b = T.shape[0]
    psi = psi_0.reshape(b, N, N)

    for jj in range(1, n_scales):
        # compute jth diffusion operator (wavelet kernel)
        T_j = torch.matrix_power(T, 2 ** (jj - 1))
        # compute jth wavelet filter
        psi_j = torch.matmul(T_j, (I_N - T_j))
        # append wavelets
        psi = torch.cat((psi, psi_j.reshape(b, N, N)), axis=0)

    psi = psi.reshape(int(psi.shape[0] / n_scales), n_scales, n_nodes, n_nodes)
    return psi

"""testing suite for kernel.py

TODO:
    - write better tests
    - test func passing for hann kernel
"""
# import torch

# from gsxform import graph, kernel

# from .test_utils import create_adj

# def test_spline_kernel():  # type: ignore
#     """Test kernel.spline_kernel for shape."""

#     # create dummy problem
#     x = torch.rand((16, 10, 10))

#     W_adj = create_adj(x)

#     E, V = graph.compute_spectra(W_adj)
#     eigs = torch.diag(E)
#     eig_max = torch.max(eigs)
#     eig_min = torch.min(eigs)
#     x1 = 1
#     x2 = 2
#     # dummy scales for kernel
#     t = torch.logspace(torch.log10(x2 / eig_min), torch.log10(x1 / eig_max), 3)

#     psi = kernel.spline_kernel(t[0] * eigs)

#     assert len(psi) == 10


# def test_hann_kernel():  # type: ignore
#     """test kernel.hann_kernel for shape"""

#     # create dummy problem
#     x = torch.rand((16, 10, 10))

#     W_adj = create_adj(x)

#     E, V = graph.compute_spectra(W_adj)
#     eigs = torch.diag(E)
#     eig_max = torch.max(eigs)
#     J = 5
#     R = 4
#     t = torch.arange(1, 4 + 1) * eig_max / (J + 1 - R)
#     psi = kernel.hann_kernel(eigs - t[0], J, R, eig_max)

#     assert len(psi) == 10

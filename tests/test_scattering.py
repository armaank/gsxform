"""testing suite for kernel.py

TODO:
    - write better tests
    - add check for output feature dim
"""
import torch

from gsxform import scattering

from .test_utils import create_adj


def test_diffusion():  # type: ignore
    """test scattering.Diffusion class"""

    # input graph, 100 nodes, batch of 16
    g = torch.rand((16, 1000, 1000))
    x = torch.rand((16, 100, 1000))  # batch_size, n_features, n_nodes
    W_adj = create_adj(g)

    # testing variations:
    for jj in [3, 4, 5]:
        for ll in [2, 3, 4]:
            txform = scattering.Diffusion(W_adj, jj, ll)
            phi = txform(x)
            #
            assert phi.shape[0:2] == torch.Size([16, 100])

            assert not torch.isnan(phi).any()


def test_tighthann():  # type: ignore
    """test scattering.TightHann class"""

    # input graph, 100 nodes, batch of 16
    g = torch.rand((16, 1000, 1000))
    x = torch.rand((16, 100, 1000))  # batch_size, n_features, n_nodes
    W_adj = create_adj(g)

    # testing variations:
    for jj in [4, 5]:  # check constrains on 2 due to R, M
        for ll in [2, 3, 4]:  # check constrain on 2 due to R and M
            txform = scattering.TightHann(W_adj, jj, ll)
            phi = txform(x)

            assert phi.shape[0:2] == torch.Size([16, 100])

            assert not torch.isnan(phi).any()


# def test_geometric():  # type: ignore
#     """test scattering.Geometric class"""

#     g = torch.rand((16, 1000, 1000))
#     x = torch.rand((16, 100, 1000))  # batch_size, n_features, n_nodes
#     W_adj = create_adj(g)

#     # testing variations:
#     for jj in [4, 5]:  # check constrains on 2 due to R, M
#         for ll in [2, 3, 4]:  # check constrain on 2 due to R and M
#             txform = scattering.Geometric(W_adj, jj, ll, 4)
#             phi = txform(x)

#             assert not torch.isnan(phi).any()

#     assert phi.shape[0:2] == torch.Size([16, 100])


# def test_warp():  # type: ignore
#     """test scattering.TightHann class w/ warping"""

#     # input graph, 100 nodes, batch of 16
#     g = torch.rand((16, 1000, 1000))
#     x = torch.rand((16, 100, 1000))  # batch_size, n_features, n_nodes
#     W_adj = create_adj(g)

#     # testing variations:
#     txform = scattering.TightHann(W_adj, 4, 3, warp=True)
#     phi = txform(x)

#     #
#     assert phi.shape[0:2] == torch.Size([16, 100])

#     assert not torch.isnan(phi).any()


# def test_spline(): # type: ignore
#    """test scattering.Spline class"""

#    # input graph, 100 nodes, batch of 16
#    g = torch.rand((16, 1000, 1000))
#    x = torch.rand((16, 100, 1000)) # batch_size, n_features, n_nodes
#    W_adj = create_adj(g)

#    # testing variations:
#    for j in [3, 4, 5]:
#        for l in [2, 3, 4]:
#            txform = scattering.Spline(W_adj,j, l)
#            phi = txform(x)
#            #
#            assert phi.shape[0:2] == torch.Size([16, 100])

"""testing suite for scattering.py

TODO:
    - add check for output feature dim
"""

import torch

from gsxform import scattering
from gsxform.graph import lazy_diffusion

from .test_utils import create_adj

torch.manual_seed(0)


def test_diffusion():  # type: ignore
    """test scattering.Diffusion class"""

    # input graph, 64 nodes, batch of 8
    g = torch.rand((8, 64, 64))
    x = torch.rand((8, 8, 64))  # batch_size, n_features, n_nodes
    W_adj = create_adj(g)

    # testing variations:
    for jj in [3, 4, 5]:
        for ll in [2, 3, 4]:
            txform = scattering.Diffusion(W_adj, jj, ll)
            phi = txform(x)
            #
            assert phi.shape[0:2] == torch.Size([8, 8])

            assert not torch.isnan(phi).any()


def test_tighthann():  # type: ignore
    """test scattering.TightHann class"""

    # input graph, 64 nodes, batch of 8
    g = torch.rand((8, 64, 64))
    x = torch.rand((8, 8, 64))  # batch_size, n_features, n_nodes
    W_adj = create_adj(g)

    # testing variations:
    for jj in [4, 5]:  # check constrains on 2 due to R, M
        for ll in [2, 3, 4]:  # check constrain on 2 due to R and M
            txform = scattering.TightHann(W_adj, jj, ll)
            phi = txform(x)

            assert phi.shape[0:2] == torch.Size([8, 8])

            assert not torch.isnan(phi).any()


def test_diffusion_batch_invariance():  # type: ignore
    """each graph's representation must not depend on its batch-mates"""

    g = torch.rand((4, 32, 32))
    x = torch.rand((4, 3, 32))
    W_adj = create_adj(g)

    batched = scattering.Diffusion(W_adj, 3, 3)(x)
    per_graph = torch.cat(
        [
            scattering.Diffusion(W_adj[ii : ii + 1], 3, 3)(x[ii : ii + 1])
            for ii in range(W_adj.shape[0])
        ],
        dim=0,
    )

    assert torch.allclose(batched, per_graph, atol=1e-6)


def test_tighthann_batch_invariance():  # type: ignore
    """each graph's representation must not depend on its batch-mates"""

    g = torch.rand((4, 32, 32))
    x = torch.rand((4, 3, 32))
    W_adj = create_adj(g)

    batched = scattering.TightHann(W_adj, 4, 3)(x)
    per_graph = torch.cat(
        [
            scattering.TightHann(W_adj[ii : ii + 1], 4, 3)(x[ii : ii + 1])
            for ii in range(W_adj.shape[0])
        ],
        dim=0,
    )

    assert torch.allclose(batched, per_graph, atol=1e-6)


def test_tighthann_small_graph():  # type: ignore
    """graphs with fewer than 10 nodes must not collapse the warp step to zero"""

    g = torch.rand((2, 8, 8))
    x = torch.rand((2, 3, 8))

    phi = scattering.TightHann(create_adj(g), 3, 2)(x)

    assert not torch.isnan(phi).any()


def test_tighthann_repeated_eigenvalues():  # type: ignore
    """disconnected and symmetric graphs give repeated eigenvalues"""

    # two disjoint triangles plus isolated nodes
    W_adj = torch.zeros((1, 12, 12))
    for aa, bb in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        W_adj[0, aa, bb] = W_adj[0, bb, aa] = 1.0

    phi = scattering.TightHann(W_adj, 3, 2)(torch.rand((1, 3, 12)))

    assert torch.isfinite(phi).all()


def test_positive_homogeneity():  # type: ignore
    """S(a x) == a S(x) for a > 0: every step is linear apart from a homogeneous nlin"""

    g = torch.rand((4, 32, 32))
    x = torch.rand((4, 3, 32))
    W_adj = create_adj(g)

    for cls in (scattering.Diffusion, scattering.TightHann):
        txform = cls(W_adj, 4, 3)

        assert torch.allclose(txform(2.5 * x), 2.5 * txform(x), atol=1e-5)


def test_non_expansive():  # type: ignore
    """the transform never increases distances, since abs is 1-Lipschitz"""

    g = torch.rand((4, 32, 32))
    x = torch.rand((4, 3, 32))
    y = torch.rand((4, 3, 32))
    W_adj = create_adj(g)

    for cls in (scattering.Diffusion, scattering.TightHann):
        txform = cls(W_adj, 4, 3)

        assert (txform(x) - txform(y)).norm() <= (x - y).norm()


def test_permutation_invariance():  # type: ignore
    """Relabelling the nodes leaves the representation unchanged. (perm equivariant)."""

    n_nodes = 32
    W_adj = create_adj(torch.rand((2, n_nodes, n_nodes)))
    x = torch.rand((2, 3, n_nodes))

    P = torch.eye(n_nodes)[torch.randperm(n_nodes)]

    for cls in (scattering.Diffusion, scattering.TightHann):
        phi = cls(W_adj, 4, 3)(x)
        phi_permuted = cls(P.matmul(W_adj).matmul(P.T), 4, 3)(x.matmul(P.T))

        assert torch.allclose(phi, phi_permuted, atol=1e-5)


def test_diffusion_stability_to_graph_perturbation():  # type: ignore
    """Perturbing the graph perturbs the representation proportionately."""

    n_nodes = 32
    W_adj = create_adj(torch.rand((1, n_nodes, n_nodes)))
    x = torch.rand((1, 3, n_nodes))

    T = lazy_diffusion(W_adj)
    phi = scattering.Diffusion(W_adj, 4, 3)(x)

    # one perturbation direction at shrinking magnitude, so the graphs nest
    S = torch.triu(torch.rand((1, n_nodes, n_nodes)), diagonal=1)
    S = S + S.transpose(-2, -1)

    distances = []
    deviations = []
    for eps in [0.4, 0.2, 0.1, 0.05, 0.025]:
        W_perturbed = (W_adj + eps * S).clamp(min=0)

        distance = torch.linalg.matrix_norm(
            T - lazy_diffusion(W_perturbed), ord=2
        ).max()
        deviation = (phi - scattering.Diffusion(W_perturbed, 4, 3)(x)).norm()

        assert deviation <= distance.sqrt() * x.norm()

        distances.append(float(distance))
        deviations.append(float(deviation))

    # shrinking the perturbation shrinks both the distance and the deviation
    assert distances == sorted(distances, reverse=True)

    assert deviations == sorted(deviations, reverse=True)


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

"""testing suite for scattering.py

TODO:
    - add check for output feature dim
"""

import pytest
import torch

from gsxform import scattering
from gsxform.graph import lazy_diffusion

from .test_utils import create_adj, pad_graphs

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
            txform = scattering.Diffusion(jj, ll)
            phi = txform(x, W_adj)
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
            txform = scattering.TightHann(jj, ll)
            phi = txform(x, W_adj)

            assert phi.shape[0:2] == torch.Size([8, 8])

            assert not torch.isnan(phi).any()


def test_diffusion_batch_invariance():  # type: ignore
    """each graph's representation must not depend on its batch-mates"""

    g = torch.rand((4, 32, 32))
    x = torch.rand((4, 3, 32))
    W_adj = create_adj(g)

    txform = scattering.Diffusion(3, 3)
    batched = txform(x, W_adj)
    per_graph = torch.cat(
        [txform(x[ii : ii + 1], W_adj[ii : ii + 1]) for ii in range(W_adj.shape[0])],
        dim=0,
    )

    assert torch.allclose(batched, per_graph, atol=1e-6)


def test_tighthann_batch_invariance():  # type: ignore
    """each graph's representation must not depend on its batch-mates"""

    g = torch.rand((4, 32, 32))
    x = torch.rand((4, 3, 32))
    W_adj = create_adj(g)

    txform = scattering.TightHann(4, 3)
    batched = txform(x, W_adj)
    per_graph = torch.cat(
        [txform(x[ii : ii + 1], W_adj[ii : ii + 1]) for ii in range(W_adj.shape[0])],
        dim=0,
    )

    assert torch.allclose(batched, per_graph, atol=1e-6)


def test_tighthann_small_graph():  # type: ignore
    """graphs with fewer than 10 nodes must not collapse the warp step to zero"""

    g = torch.rand((2, 8, 8))
    x = torch.rand((2, 3, 8))

    phi = scattering.TightHann(3, 2)(x, create_adj(g))

    assert not torch.isnan(phi).any()


def test_tighthann_repeated_eigenvalues():  # type: ignore
    """disconnected and symmetric graphs give repeated eigenvalues"""

    # two disjoint triangles plus isolated nodes
    W_adj = torch.zeros((1, 12, 12))
    for aa, bb in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        W_adj[0, aa, bb] = W_adj[0, bb, aa] = 1.0

    phi = scattering.TightHann(3, 2)(torch.rand((1, 3, 12)), W_adj)

    assert torch.isfinite(phi).all()


def test_positive_homogeneity():  # type: ignore
    """S(a x) == a S(x) for a > 0: every step is linear apart from a homogeneous nlin"""

    g = torch.rand((4, 32, 32))
    x = torch.rand((4, 3, 32))
    W_adj = create_adj(g)

    for cls in (scattering.Diffusion, scattering.TightHann):
        txform = cls(4, 3, cached=True)

        assert torch.allclose(txform(2.5 * x, W_adj), 2.5 * txform(x, W_adj), atol=1e-5)


def test_non_expansive():  # type: ignore
    """the transform never increases distances, since abs is 1-Lipschitz"""

    g = torch.rand((4, 32, 32))
    x = torch.rand((4, 3, 32))
    y = torch.rand((4, 3, 32))
    W_adj = create_adj(g)

    for cls in (scattering.Diffusion, scattering.TightHann):
        txform = cls(4, 3, cached=True)

        assert (txform(x, W_adj) - txform(y, W_adj)).norm() <= (x - y).norm()


def test_permutation_invariance():  # type: ignore
    """Relabelling the nodes leaves the representation unchanged. (perm equivariant)."""

    n_nodes = 32
    W_adj = create_adj(torch.rand((2, n_nodes, n_nodes)))
    x = torch.rand((2, 3, n_nodes))

    P = torch.eye(n_nodes)[torch.randperm(n_nodes)]

    for cls in (scattering.Diffusion, scattering.TightHann):
        txform = cls(4, 3)
        phi = txform(x, W_adj)
        phi_permuted = txform(x.matmul(P.T), P.matmul(W_adj).matmul(P.T))

        assert torch.allclose(phi, phi_permuted, atol=1e-5)


def test_diffusion_stability_to_graph_perturbation():  # type: ignore
    """Perturbing the graph perturbs the representation proportionately."""

    n_nodes = 32
    W_adj = create_adj(torch.rand((1, n_nodes, n_nodes)))
    x = torch.rand((1, 3, n_nodes))

    T = lazy_diffusion(W_adj)
    txform = scattering.Diffusion(4, 3)
    phi = txform(x, W_adj)

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
        deviation = (phi - txform(x, W_perturbed)).norm()

        assert deviation <= distance.sqrt() * x.norm()

        distances.append(float(distance))
        deviations.append(float(deviation))

    # shrinking the perturbation shrinks both the distance and the deviation
    assert distances == sorted(distances, reverse=True)

    assert deviations == sorted(deviations, reverse=True)


def test_padded_batch_matches_per_graph():  # type: ignore
    """A masked, padded batch of mixed-size graphs equals running each alone."""

    torch.manual_seed(0)

    cases = [
        ([9, 17, 24, 12], scattering.Diffusion(4, 3)),
        ([20, 32, 26, 40], scattering.TightHann(4, 3)),
        ([20, 32, 26, 40], scattering.TightHann(4, 3, use_warp=False)),
    ]
    for sizes, txform in cases:
        graphs, signals, W_adj, x, mask = pad_graphs(sizes)
        per_graph = torch.cat(
            [txform(s[None], g[None]) for g, s in zip(graphs, signals, strict=True)]
        )

        assert torch.allclose(txform(x, W_adj, mask), per_graph, atol=1e-5)


def test_padding_contents_ignored():  # type: ignore
    """Whatever fills the padding, the masked output is unchanged."""

    _, _, W_adj, x, mask = pad_graphs([9, 17, 24, 12])

    pad_nodes = ~mask
    pad_edges = ~(mask[:, :, None] & mask[:, None, :])
    x_noisy = torch.where(pad_nodes[:, None, :], torch.rand_like(x), x)
    W_noisy = torch.where(pad_edges, torch.rand_like(W_adj), W_adj)

    for cls in (scattering.Diffusion, scattering.TightHann):
        txform = cls(4, 3)

        assert torch.allclose(
            txform(x_noisy, W_noisy, mask), txform(x, W_adj, mask), atol=1e-5
        )


def test_cache_keyed_on_mask():  # type: ignore
    """With cached=True, changing only the mask rebuilds the bank."""

    graphs, signals, W_adj, x, mask = pad_graphs([9, 24])
    txform = scattering.TightHann(4, 3, cached=True)

    txform(x, W_adj, torch.ones_like(mask))
    masked = txform(x, W_adj, mask)

    assert torch.allclose(masked, scattering.TightHann(4, 3)(x, W_adj, mask))


def test_path_index():  # type: ignore
    """path_index lists paths depth-major, then by parent, then by scale."""

    assert scattering.Diffusion(2, 3).path_index() == [
        (),
        (0,),
        (1,),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]

    W_adj = create_adj(torch.rand((2, 16, 16)))
    x = torch.rand((2, 3, 16))
    for n_scales, n_layers in [(3, 2), (4, 3), (3, 4)]:
        txform = scattering.Diffusion(n_scales, n_layers)

        assert txform(x, W_adj).shape[-1] == len(txform.path_index())


def test_path_index_matches_coefficients():  # type: ignore
    """Each coefficient is the mean of |psi_jl ... |psi_j1 x|| along its path."""

    W_adj = create_adj(torch.rand((2, 16, 16)))
    x = torch.rand((2, 3, 16))
    txform = scattering.Diffusion(3, 3)
    psi = txform.get_wavelets(W_adj)

    phi = txform(x, W_adj)
    for cc, path in enumerate(txform.path_index()):
        field = x
        for jj in path:
            field = torch.abs(field.matmul(psi[:, jj]))

        assert torch.allclose(phi[..., cc], field.mean(dim=-1), atol=1e-6)


def test_node_output_pools_to_graph_output():  # type: ignore
    """Averaging node output over nodes gives graph output."""

    _, _, W_adj, x, mask = pad_graphs([9, 17, 24, 12])

    for cls in (scattering.Diffusion, scattering.TightHann):
        nodes = cls(4, 3, output="node")(x, W_adj, mask)
        graph = cls(4, 3)(x, W_adj, mask)

        n_real = mask.sum(dim=-1)[:, None, None]
        assert nodes.shape == (4, 3, 21, 24)
        assert torch.allclose(nodes.sum(dim=-1) / n_real, graph, atol=1e-6)
        # padded nodes are zeroed
        assert (nodes.masked_select(~mask[:, None, None, :]) == 0).all()


def test_node_output_equivariant():  # type: ignore
    """Relabelling the nodes relabels node output the same way."""

    n_nodes = 32
    W_adj = create_adj(torch.rand((2, n_nodes, n_nodes)))
    x = torch.rand((2, 3, n_nodes))
    P = torch.eye(n_nodes)[torch.randperm(n_nodes)]

    for cls in (scattering.Diffusion, scattering.TightHann):
        txform = cls(4, 3, output="node")

        nodes = txform(x, W_adj)
        nodes_permuted = txform(x.matmul(P.T), P.matmul(W_adj).matmul(P.T))

        assert torch.allclose(nodes.matmul(P.T), nodes_permuted, atol=1e-5)


def test_bad_options_raise():  # type: ignore
    """Unknown output or aggregation modes are rejected at construction."""

    with pytest.raises(ValueError, match="output"):
        scattering.Diffusion(4, 3, output="edge")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="aggregation"):
        scattering.Diffusion(4, 3, aggregation="max")  # type: ignore[arg-type]


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

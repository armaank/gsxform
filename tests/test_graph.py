"""testing suite for graph.py"""

import torch

from gsxform import graph

from .test_utils import create_adj

torch.manual_seed(0)


def test_adjacency_to_laplacian():  # type: ignore
    """Test graph.adjacency_to_laplacian."""

    x = torch.rand((8, 16, 16))

    W_adj = create_adj(x)

    L = graph.adjacency_to_laplacian(W_adj)

    assert L.size() == W_adj.size()


def test_normalize_adjacency():  # type: ignore
    """Test graph.normalize_adjacency for shape and numerical stability."""

    x = torch.rand((8, 16, 16))

    W_adj = create_adj(x)

    W_norm = graph.normalize_adjacency(W_adj)

    assert W_adj.size() == W_norm.size()

    is_nan = torch.isnan(W_norm)

    assert not torch.any(is_nan)


def test_normalize_laplacian():  # type: ignore
    """Test graph.normalize_adjacency for shape and numerical stability."""

    x = torch.rand((8, 16, 16))

    W_adj = create_adj(x)

    L = graph.adjacency_to_laplacian(W_adj)

    L_norm = graph.normalize_laplacian(L)

    assert L.size() == L_norm.size()

    is_nan = torch.isnan(L_norm)

    assert not torch.any(is_nan)


def test_compute_spectra():  # type: ignore
    """Test graph.compute_spectra for shape."""

    x = torch.rand((8, 16, 16))

    W_adj = create_adj(x)

    E, V = graph.compute_spectra(W_adj)

    assert E.size() == (8, 16)

    assert V.size() == x.size()


def test_laplacian_row_sums_vanish():  # type: ignore
    """Each row of L sums to zero: the degree on the diagonal cancels the row."""

    W_adj = create_adj(torch.rand((8, 16, 16)))

    L = graph.adjacency_to_laplacian(W_adj)

    assert torch.allclose(L.sum(-1), torch.zeros(8, 16), atol=1e-6)


def test_laplacian_is_symmetric():  # type: ignore
    """An undirected graph gives a symmetric Laplacian."""

    W_adj = create_adj(torch.rand((8, 16, 16)))

    L = graph.adjacency_to_laplacian(W_adj)

    assert torch.allclose(L, L.transpose(-2, -1), atol=1e-6)


def test_compute_spectra_eigenvectors_orthonormal():  # type: ignore
    """Eigenvectors of the symmetric normalized Laplacian form an orthonormal basis."""

    W_adj = create_adj(torch.rand((8, 16, 16)))

    _, V = graph.compute_spectra(W_adj)

    identity = torch.eye(16).expand(8, 16, 16)

    assert torch.allclose(V.matmul(V.transpose(-2, -1)), identity, atol=1e-5)

    assert torch.allclose(V.transpose(-2, -1).matmul(V), identity, atol=1e-5)


def test_compute_spectra_reconstructs_laplacian():  # type: ignore
    """V diag(E) V^T recovers the normalized Laplacian it was decomposed from."""

    W_adj = create_adj(torch.rand((8, 16, 16)))

    L_norm = graph.normalize_laplacian(graph.adjacency_to_laplacian(W_adj))
    E, V = graph.compute_spectra(W_adj)

    recon = V.matmul(torch.diag_embed(E)).matmul(V.transpose(-2, -1))

    assert torch.allclose(recon, L_norm, atol=1e-5)


def test_lazy_diffusion():  # type: ignore
    """T is 1/2 (I + D^-1/2 W D^-1/2), Gama et. al 2018 eq. (3) and sec. 3.3."""

    W_adj = create_adj(torch.rand((4, 16, 16)))

    degree = W_adj.sum(-1)
    # normalize_adjacency floors the degree at one; keep that path inert so the
    # assertion below is the textbook formula and nothing else
    assert (degree >= 1).all()

    D_invsqrt = torch.diag_embed(degree.pow(-0.5))
    A = D_invsqrt.matmul(W_adj).matmul(D_invsqrt)

    expected = 0.5 * (torch.eye(16) + A)

    assert torch.allclose(graph.lazy_diffusion(W_adj), expected, atol=1e-6)


def test_compute_spectra_eigenvalues_sorted_and_bounded():  # type: ignore
    """Eigenvalues come out ascending and inside [0, 2], as a normalized Laplacian must."""

    W_adj = create_adj(torch.rand((8, 16, 16)))

    E, _ = graph.compute_spectra(W_adj)

    assert (E[:, 1:] >= E[:, :-1] - 1e-6).all()

    assert (E >= -1e-6).all()

    assert (E <= 2.0 + 1e-6).all()

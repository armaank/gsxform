"""testing suite for wavelets.py"""

import torch

from gsxform.graph import compute_spectra, lazy_diffusion
from gsxform.kernel import TightHannKernel
from gsxform.wavelets import diffusion_wavelets, tighthann_wavelets

from .test_utils import create_adj, sqrt_degree

torch.manual_seed(0)

# Shuman et. al 2015, Corollary 2, for the Hann case used here; see test_kernel.py
FRAME_CONSTANT = 9 / 8


def build_tighthann(W_adj: torch.Tensor, n_scales: int) -> torch.Tensor:
    """Build a tight Hann filter bank directly, without the spectrum-adaptive warp."""
    E, _ = compute_spectra(W_adj)
    kernel = TightHannKernel(n_scales, E.max(-1).values)

    return tighthann_wavelets(W_adj, n_scales, kernel)


def littlewood_paley(psi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Total energy the filter bank extracts from x, relative to the energy of x."""
    energy = sum(
        (torch.einsum("bnm,bm->bn", psi[:, jj], x) ** 2).sum(-1)
        for jj in range(psi.shape[1])
    )

    return energy / (x**2).sum(-1)


def test_diffusion_wavelets_telescope():  # type: ignore
    """The filter bank telescopes to I - T^(2^(J-1))."""

    for n_scales in [2, 3, 4, 5, 6]:
        W_adj = create_adj(torch.rand((2, 24, 24)))
        T = lazy_diffusion(W_adj)

        psi = diffusion_wavelets(T, n_scales)

        expected = torch.eye(24) - torch.matrix_power(T, 2 ** (n_scales - 1))

        assert torch.allclose(psi.sum(1), expected, atol=1e-5)


def test_diffusion_wavelets_littlewood_paley():  # type: ignore
    """The bank does not amplify: sum_j ||psi_j x||^2 <= ||x||^2."""

    for n_nodes in [16, 32, 64]:
        for n_scales in [3, 4, 5]:
            W_adj = create_adj(torch.rand((4, n_nodes, n_nodes)))
            psi = diffusion_wavelets(lazy_diffusion(W_adj), n_scales)

            v = sqrt_degree(W_adj)
            x = torch.rand((4, n_nodes))
            # the bound holds on the orthogonal complement of v
            x = x - (x * v).sum(-1, keepdim=True) * v

            assert (littlewood_paley(psi, x) <= 1.0 + 1e-5).all()


def test_tighthann_wavelets_shape():  # type: ignore
    """One filter per scale, each an operator over the graph."""

    W_adj = create_adj(torch.rand((3, 16, 16)))

    psi = build_tighthann(W_adj, 4)

    assert psi.shape == torch.Size([3, 4, 16, 16])

    assert not torch.isnan(psi).any()


def test_tighthann_wavelets_symmetric():  # type: ignore
    """Each filter is V diag(g_j(E)) V^T, so it inherits the Laplacian's symmetry."""

    W_adj = create_adj(torch.rand((3, 16, 16)))

    psi = build_tighthann(W_adj, 4)

    assert torch.allclose(psi, psi.transpose(-2, -1), atol=1e-5)


def test_tighthann_wavelets_frame_bounds():  # type: ignore
    """The spectral frame bounds carry over to the operators built from them."""

    for n_nodes in [16, 32]:
        for n_scales in [3, 4, 5]:
            W_adj = create_adj(torch.rand((3, n_nodes, n_nodes)))
            psi = build_tighthann(W_adj, n_scales)

            response = sum(psi[:, jj].matmul(psi[:, jj]) for jj in range(n_scales))
            eigenvalues = torch.linalg.eigvalsh(response)

            assert eigenvalues.min() >= FRAME_CONSTANT / 2 - 1e-4

            assert eigenvalues.max() <= FRAME_CONSTANT + 1e-4


def test_tighthann_wavelets_littlewood_paley():  # type: ignore
    """Energy extracted from any signal stays inside the same frame bounds."""

    for n_nodes in [16, 32]:
        for n_scales in [3, 4, 5]:
            W_adj = create_adj(torch.rand((3, n_nodes, n_nodes)))
            psi = build_tighthann(W_adj, n_scales)

            ratio = littlewood_paley(psi, torch.rand((3, n_nodes)))

            assert (ratio >= FRAME_CONSTANT / 2 - 1e-4).all()

            assert (ratio <= FRAME_CONSTANT + 1e-4).all()

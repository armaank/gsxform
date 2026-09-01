"""testing suite for kernel.py

TODO:
    - test func passing for hann kernel
"""

import pytest
import torch

from gsxform.kernel import TightHannKernel


def test_default_omega():  # type: ignore
    """omega is documented as optional, so the default must be usable"""

    kernel = TightHannKernel(3, torch.tensor(2.0))

    adapted_kernel = kernel.get_adapted_kernel(torch.linspace(0, 2, 10), 1)

    assert torch.isfinite(adapted_kernel).all()


def test_rejects_degenerate_n_scales():  # type: ignore
    """n_scales at or below R - 1 makes the dilation factor blow up"""

    with pytest.raises(ValueError, match="n_scales"):
        TightHannKernel(2, torch.tensor(2.0))


def test_per_graph_max_eig():  # type: ignore
    """a max_eig per graph broadcasts against batched eigenvalues"""

    kernel = TightHannKernel(3, torch.tensor([2.0, 1.5]))

    adapted_kernel = kernel.get_adapted_kernel(torch.rand((2, 10)), 1)

    assert adapted_kernel.shape == torch.Size([2, 10])


# Shuman et. al 2015, Corollary 2: where the translates fully overlap, the squared
# responses sum to R*a_0^2 + (R/2)*sum_k a_k^2. This implementation uses the Hann
# case, K = 1 and a_0 = a_1 = 1/2, with R = 3.
FRAME_CONSTANT = 9 / 8


def test_frame_constant_across_the_covered_band():  # type: ignore
    """Where R translates overlap, the squared responses sum to the Corollary 2 constant."""

    for n_scales in [4, 5, 6]:
        kernel = TightHannKernel(n_scales, torch.tensor(2.0))

        # the top of the band is always fully covered; the low end is not, see below
        eig = torch.linspace(1.5, 2.0, 200)
        response = sum(
            kernel.get_adapted_kernel(eig, scale + 1) ** 2 for scale in range(n_scales)
        )

        assert torch.allclose(
            response, torch.full_like(response, FRAME_CONSTANT), atol=1e-4
        )


def test_frame_bounds():  # type: ignore
    """Test Hann frame bounds."""

    for n_scales in [3, 4, 5, 6]:
        for max_eig in [1.0, 1.5, 2.0]:
            kernel = TightHannKernel(n_scales, torch.tensor(max_eig))

            eig = torch.linspace(0, max_eig, 500)
            response = sum(
                kernel.get_adapted_kernel(eig, scale + 1) ** 2
                for scale in range(n_scales)
            )

            assert response.min() >= FRAME_CONSTANT / 2 - 1e-5

            assert response.max() <= FRAME_CONSTANT + 1e-5

            # the deficit sits at the bottom of the spectrum, and is exactly half
            assert torch.isclose(
                response[0], torch.tensor(FRAME_CONSTANT / 2), atol=1e-5
            )


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

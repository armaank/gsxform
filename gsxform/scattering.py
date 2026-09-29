"""Generic base classes for scattering transform operations.

TODO:
    - confirm symbolic notation
"""

from collections.abc import Callable
from functools import partial

import torch
from einops import einsum, rearrange, repeat
from torch import nn

from .graph import compute_spectra, lazy_diffusion
from .kernel import TightHannKernel
from .wavelets import diffusion_wavelets, tighthann_wavelets


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Interpolate linearly between reference points, one set per graph.

    Equivalent to `scipy.interpolate.interp1d(..., fill_value="extrapolate")`
    applied to each graph, but stays in torch, so it works on GPU and under
    autograd. See [1] and [2].

    Parameters
    ----------
    x: torch.Tensor
        Query points, shaped (batch, n_queries) or (batch,)
    xp: torch.Tensor
        Sorted reference x-coordinates, shaped (batch, n_points)
    fp: torch.Tensor
        Reference y-coordinates, shaped (batch, n_points)

    Returns
    -------
    torch.Tensor
        Interpolated values, shaped like `x`

    References
    ----------
    .. [1] https://github.com/xitorch/xitorch/blob/master/xitorch/_impls/interpolate/interp_1d.py
    .. [2] https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py

    """
    # a single query per graph still needs the axis searchsorted expects
    squeeze = x.ndim < xp.ndim
    if squeeze:
        x = x.unsqueeze(-1)

    # clamping to an interior index is what extrapolates: queries off either end
    # reuse the slope of the first or last segment
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.shape[-1] - 1)

    x0, x1 = torch.gather(xp, -1, idx - 1), torch.gather(xp, -1, idx)
    y0, y1 = torch.gather(fp, -1, idx - 1), torch.gather(fp, -1, idx)

    # graph spectra repeat eigenvalues whenever the graph is disconnected or
    # symmetric, which collapses a segment to zero width
    dx = (x1 - x0).clamp(min=torch.finfo(xp.dtype).eps)

    out = y0 + (y1 - y0) / dx * (x - x0)

    return out.squeeze(-1) if squeeze else out


class ScatteringTransform(nn.Module):
    """ScatteringTransform base class. Inherits from PyTorch nn.Module.

    This class implements the base logic to compute graph scattering
    transforms with a pooling and an arbitrary wavelet transform
    operators.

    """

    _cached_adj: torch.Tensor | None
    _cached_psi: torch.Tensor | None

    def __init__(
        self,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
        cached: bool = False,
    ) -> None:
        """Initialize scattering transform base class.

        This is a base class, and implements only the logic to compute
        an arbitrary scattering transform. The method `get_wavelets`
        must be implemented by the subclass

        Parameters
        ----------
        n_scales: int
            Number of scales to use in wavelet transform
        n_layers: int
            Number of layers in the scattering transform
        nlin: Callable
            Non-linearity used in the scattering transform. Defaults to torch.abs
        cached: bool
            If set to True, the filter bank is cached on first execution and
            reused while the same adjacency matrix is passed. A different
            adjacency, or one that requires grad, rebuilds it. Defaults to False
        """
        super().__init__()

        # number of scales
        self.n_scales = n_scales
        # number of layers
        self.n_layers = n_layers

        self.nlin = nlin

        self.cached = cached

        # derived from the graph and shaped by its size, so kept out of the
        # state dict; the adjacency is kept to detect when the graph changes
        self.register_buffer("_cached_adj", None, persistent=False)
        self.register_buffer("_cached_psi", None, persistent=False)

    def reset_parameters(self) -> None:
        """Clear the cached filter bank."""
        self._cached_adj = None
        self._cached_psi = None

    def _build_wavelets(self, W_adj: torch.Tensor) -> torch.Tensor:
        """Check the adjacency matrix is square, then build its filter bank."""
        assert W_adj.shape[-1] == W_adj.shape[-2], (
            f"W_adj must be square, got shape {tuple(W_adj.shape)}"
        )
        return self.get_wavelets(W_adj)

    def _get_wavelets_cached(self, W_adj: torch.Tensor) -> torch.Tensor:
        """Return the filter bank for `W_adj`, reusing the cache when enabled."""
        
        if not self.cached or W_adj.requires_grad:
            return self._build_wavelets(W_adj)

        cached_adj = self._cached_adj
        if (
            self._cached_psi is not None
            and cached_adj is not None
            and cached_adj.shape == W_adj.shape
            and cached_adj.dtype == W_adj.dtype
            and cached_adj.device == W_adj.device
            and torch.equal(cached_adj, W_adj)
        ):
            return self._cached_psi

        psi = self._build_wavelets(W_adj)
        self._cached_adj = W_adj.detach().clone()
        self._cached_psi = psi
        return psi

    def get_wavelets(self, W_adj: torch.Tensor) -> torch.Tensor:
        """Compute the wavelet operator.

        Subclasses are required to implement this method.

        Parameters
        ----------
        W_adj: torch.Tensor
            Batch of weighted adjacency matrices
        """
        raise NotImplementedError

    def get_lowpass(
        self,
        batch_size: int,
        n_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute lowpass filtering/pooling operator.

        This should roughly resemble an average, it alters the output
        scaling factor. For instance averaging with the norm
        of the degree vector scales towards zero, this implementation
        offers a more natural scaling.

        Parameters
        ----------
        batch_size: int
            Number of graphs in the batch
        n_nodes: int
            Number of nodes per graph
        device: torch.device
            Device to allocate the operator on
        dtype: torch.dtype
            Floating point type of the operator

        Returns
        -------
        lowpass: torch.Tensor
            average pooling operator
        """
        lowpass = (1 / n_nodes) * torch.ones(
            batch_size, n_nodes, device=device, dtype=dtype
        )

        return lowpass

    def forward(self, x: torch.Tensor, W_adj: torch.Tensor) -> torch.Tensor:
        """Forward pass of a generic scattering transform.

        Parameters
        ----------
        x: torch.Tensor
            input batch of graph signals
        W_adj: torch.Tensor
            Batch of weighted adjacency matrices

        Returns
        -------
        phi: torch.Tensor
            scattering representation of the input batch

        """
        batch_size, n_features, n_nodes = x.shape

        psi = self._get_wavelets_cached(W_adj)

        assert psi.shape[0] == batch_size, (
            f"batch size of x ({batch_size}) must match W_adj ({psi.shape[0]})"
        )
        assert psi.shape[-1] == n_nodes, (
            f"node count of x ({n_nodes}) must match W_adj ({psi.shape[-1]})"
        )

        lowpass = self.get_lowpass(batch_size, n_nodes, x.device, x.dtype)

        # compute first scattering layer, low pass filter
        phi: torch.Tensor = einsum(x, lowpass, "b f n, b n -> b f")
        phi = rearrange(phi, "b f -> b f 1")

        # reshape inputs for loop
        S_x = rearrange(x, "b f n -> b 1 f n")

        for ll in range(1, self.n_layers):
            S_x_ll = torch.empty(
                [batch_size, 0, n_features, n_nodes], device=x.device, dtype=x.dtype
            )

            for jj in range(self.n_scales ** (ll - 1)):
                # intermediate repr, one copy per scale to contract against psi
                x_jj = repeat(S_x[:, jj, :, :], "b f n -> b ns f n", ns=self.n_scales)

                # wavelet filtering operation
                psi_x_jj = einsum(x_jj, psi, "b ns f n, b ns n m -> b ns f m")

                # application of non-linearity, yields scattering output
                S_x_jj = self.nlin(psi_x_jj)

                # concat scattering scale for the layer
                S_x_ll = torch.cat((S_x_ll, S_x_jj), dim=1)

                # compute scattering representation
                phi_jj = einsum(S_x_jj, lowpass, "b ns f n, b n -> b f ns")

                phi = torch.cat((phi, phi_jj), dim=2)

            S_x = S_x_ll.clone()  # continue iteration through the layer

        return phi


class Diffusion(ScatteringTransform):
    """Diffusion scattering transform.

    Subclass of `ScatteringTransform`, implements `get_wavelets` method.
    Diffusion scattering transform algorithm based on description
    in Gama et. al 2018.

    """

    def __init__(
        self,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
        cached: bool = False,
    ) -> None:
        """Initialize diffusion scattering transform.

        Parameters
        ----------
        n_scales: int
            Number of scales to use in wavelet transform
        n_layers: int
            Number of layers in the scattering transform
        nlin: Callable[torch.Tensor]
            Non-linearity used in the scattering transform. Defaults to torch.abs
        cached: bool
            Cache the filter bank while the adjacency is unchanged. Defaults to
            False

        """
        super().__init__(n_scales, n_layers, nlin, cached)

    def get_wavelets(self, W_adj: torch.Tensor) -> torch.Tensor:
        """Subclass method used to get wavelet filter bank.

        This method returns diffusion wavelets

        Parameters
        ----------
        W_adj: torch.Tensor
            Batch of weighted adjacency matrices

        Returns
        -------
        psi: torch.Tensor
            diffusion wavelet operator

        """
        # compute diffusion matrix
        T = lazy_diffusion(W_adj)
        # compute wavelet operator
        psi = diffusion_wavelets(T, self.n_scales)

        return psi


class TightHann(ScatteringTransform):
    """TightHann scattering transform.

    Subclass of `ScatteringTransform`, implements `get_wavelets` methods.
    Also additionally implements functions used to compute spectrum-adaptive
    wavelets.

    """

    def __init__(
        self,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
        use_warp: bool = True,
        cached: bool = False,
    ) -> None:
        """Initialize tight Hann scattering transform.

        Parameters
        ----------
        n_scales: int
            Number of scales to use in wavelet transform
        n_layers: int
            Number of layers in the scattering transform
        nlin: Callable[torch.Tensor]
            Non-linearity used in the scattering transform. Defaults to torch.abs
        use_warp: bool
            Use warping function. Defaults to True
        cached: bool
            Cache the filter bank while the adjacency is unchanged. Defaults to
            False

        """
        super().__init__(n_scales, n_layers, nlin, cached)
        self.use_warp = use_warp

    def warp_func(
        self, spectra: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Compute the spectrum-adaptive warping function.

        Parameters
        ----------
        spectra: torch.Tensor
            Eigenvalues of each graph's Laplacian, sorted ascending, shaped
            (batch, n_nodes)

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            Per-graph piecewise linear approximation of the spectral CDF
        """
        n_eigs = spectra.shape[-1]
        cdf = torch.arange(0, n_eigs, device=spectra.device, dtype=spectra.dtype) / (
            n_eigs - 1.0
        )
        cdf = cdf.expand_as(spectra)

        step = max(1, int(n_eigs / 5 - 1))

        if self.use_warp:
            xp, fp = spectra[:, 0::step], cdf[:, 0::step]
        else:
            xp, fp = spectra, cdf

        # switched from lambda to partial so the kernel stays picklable
        return partial(_interp, xp=xp.contiguous(), fp=fp.contiguous())

    def get_kernel(self, E: torch.Tensor) -> TightHannKernel:
        """Compute TightHann kernel adaptively.

        Parameters
        ----------
        E: torch.Tensor
            Eigenvalues of each graph's Laplacian, shaped (batch, n_nodes)

        Returns
        -------
        TightHannKernel
            Kernel adapted to each graph's spectrum
        """
        # sort within each graph
        spectra, _ = torch.sort(E, dim=-1)
        return TightHannKernel(self.n_scales, spectra[:, -1], self.warp_func(spectra))

    def get_wavelets(self, W_adj: torch.Tensor) -> torch.Tensor:
        """Subclass method used to get wavelet filter bank.

        This method returns tight Hann wavelets

        Parameters
        ----------
        W_adj: torch.Tensor
            Batch of weighted adjacency matrices

        Returns
        -------
        psi: torch.Tensor
            tight Hann wavelet operator

        """

        E, V = compute_spectra(W_adj)
        psi = tighthann_wavelets(
            W_adj, self.n_scales, self.get_kernel(E), spectra=(E, V)
        )

        return psi

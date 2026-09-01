"""Generic base classes for scattering transform operations.

TODO:
    - confirm symbolic notation
"""

from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange, repeat
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

    def __init__(
        self,
        W_adj: torch.Tensor,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
        **kwargs: Any,
    ) -> None:
        """Initialize scattering transform base class.

        This is a base class, and implements only the logic to compute
        an arbitrary scattering transform. The method `get_wavelets`
        must be implemented by the subclass

        Parameters
        ----------
        W_adj: torch.Tensor
            Weighted adjacency matrix
        n_scales: int
            Number of scales to use in wavelet transform
        n_layers: int
            Number of layers in the scattering transform
        nlin: Callable
            Non-linearity used in the scattering transform. Defaults to torch.abs
        **kwargs: Any
            Additional keyword arguments
        """
        super().__init__()

        # adjacency matrix, registered so .to(device) moves it with the module
        self.register_buffer("W_adj", W_adj)
        # number of scales
        self.n_scales = n_scales
        # number of layers
        self.n_layers = n_layers

        self.n_nodes = W_adj.shape[1]
        assert W_adj.shape[1] == W_adj.shape[2]

        self.nlin = nlin

    def get_wavelets(self) -> torch.Tensor:
        """Compute the wavelet operator.

        Subclasses are required to implement this method.
        """
        raise NotImplementedError

    def get_lowpass(self, batch_size: int) -> torch.Tensor:
        """Compute lowpass filtering/pooling operator.

        This should roughly resemble an average, it alters the output
        scaling factor. For instance averaging with the norm
        of the degree vector scales towards zero, this implementation
        offers a more natural scaling.

        Parameters
        ----------
        batch_size: int
            Number of graphs in the batch

        Returns
        -------
        lowpass: torch.Tensor
            average pooling operator
        """
        lowpass = (1 / self.n_nodes) * torch.ones(
            batch_size, self.n_nodes, device=self.W_adj.device
        )

        lowpass = rearrange(lowpass, "b ni -> b ni 1")

        return lowpass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a generic scattering transform.

        Parameters
        ----------
        x: torch.Tensor
            input batch of graph signals

        Returns
        -------
        phi: torch.Tensor
            scattering representation of the input batch

        """
        batch_size = x.shape[0]

        n_features = x.shape[1]

        assert batch_size == self.W_adj.shape[0], (
            f"batch size of x ({batch_size}) must match W_adj ({self.W_adj.shape[0]})"
        )

        lowpass = self.get_lowpass(batch_size)
        psi = self.get_wavelets()

        # compute first scattering layer, low pass filter via matmul
        phi = torch.matmul(x, lowpass)

        # reshape inputs for loop
        S_x = rearrange(x, "b f n -> b 1 f n")
        lowpass = rearrange(lowpass, "b n 1 -> b 1 n 1")
        lowpass = repeat(lowpass, "b 1 n 1 -> b (1 ns) n 1", ns=self.n_scales)

        for ll in range(1, self.n_layers):
            S_x_ll = torch.empty(
                [batch_size, 0, n_features, self.n_nodes], device=x.device
            )

            for jj in range(self.n_scales ** (ll - 1)):
                # intermediate repr
                x_jj = rearrange(S_x[:, jj, :, :], "b f n -> b 1 f n")

                # wavelet filtering operation, matrix multiply
                psi_x_jj = torch.matmul(x_jj, psi)

                # application of non-linearity, yields scattering output
                S_x_jj = self.nlin(psi_x_jj)

                # concat scattering scale for the layer
                S_x_ll = torch.cat((S_x_ll, S_x_jj), dim=1)

                # compute scattering representation, matrix multiply
                phi_jj = torch.matmul(S_x_jj, lowpass)
                phi_jj = rearrange(phi_jj, "b l f 1 -> b f l")

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
        W_adj: torch.Tensor,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
    ) -> None:
        """Initialize diffusion scattering transform.

        Parameters
        ----------
        W_adj: torch.Tensor
            Weighted adjacency matrix
        n_scales: int
            Number of scales to use in wavelet transform
        n_layers: int
            Number of layers in the scattering transform
        nlin: Callable[torch.Tensor]
            Non-linearity used in the scattering transform. Defaults to torch.abs

        """
        super().__init__(W_adj, n_scales, n_layers, nlin)

    def get_wavelets(self) -> torch.Tensor:
        """Subclass method used to get wavelet filter bank.

        This method returns diffusion wavelets

        Returns
        -------
        psi: torch.Tensor
            diffusion wavelet operator

        """
        # compute diffusion matrix
        T = lazy_diffusion(self.W_adj)
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
        W_adj: torch.Tensor,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
        use_warp: bool = True,
    ) -> None:
        """Initialize tight Hann scattering transform.

        Parameters
        ----------
        W_adj: torch.Tensor
            Weighted adjacency matrix
        n_scales: int
            Number of scales to use in wavelet transform
        n_layers: int
            Number of layers in the scattering transform
        nlin: Callable[torch.Tensor]
            Non-linearity used in the scattering transform. Defaults to torch.abs
        use_warp: bool
            Use warping function. Defaults to True

        """
        super().__init__(W_adj, n_scales, n_layers, nlin)
        self.use_warp = use_warp
        self.warp = self.warp_func()

    def warp_func(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Compute the spectrum-adaptive warping function."""
        E, V = compute_spectra(self.W_adj)
        # sort within each graph
        self.spectra, _ = torch.sort(E, dim=-1)
        self.max_eig = self.spectra[:, -1]

        n_eigs = self.spectra.shape[-1]
        cdf = torch.arange(
            0, n_eigs, device=self.spectra.device, dtype=self.spectra.dtype
        ) / (n_eigs - 1.0)
        cdf = cdf.expand_as(self.spectra)

        step = max(1, int(n_eigs / 5 - 1))

        if self.use_warp:
            xp, fp = self.spectra[:, 0::step], cdf[:, 0::step]
        else:
            xp, fp = self.spectra, cdf

        self.register_buffer("_warp_xp", xp.contiguous())
        self.register_buffer("_warp_fp", fp.contiguous())

        return lambda eig: _interp(eig, self._warp_xp, self._warp_fp)

    def get_kernel(self) -> TightHannKernel:
        """Compute TightHann kernel adaptively."""
        return TightHannKernel(self.n_scales, self.max_eig, self.warp)

    def get_wavelets(self) -> torch.Tensor:
        """Subclass method used to get wavelet filter bank.

        This method returns diffusion wavelets

        Returns
        -------
        psi: torch.Tensor
            diffusion wavelet operator

        """
        # compute wavelet operator
        psi = tighthann_wavelets(self.W_adj, self.n_scales, self.get_kernel())

        return psi

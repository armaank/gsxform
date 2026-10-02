"""Generic base classes for scattering transform operations.

TODO:
    - confirm symbolic notation
"""

from collections.abc import Callable
from functools import partial
from typing import Literal

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


def _left_pad(points: torch.Tensor, width: int) -> torch.Tensor:
    """Pad (1, n) points to (1, width) by repeating the first point."""
    first = points[:, :1].expand(-1, width - points.shape[-1])
    return torch.cat((first, points), dim=-1)


def _same_tensor(cached: torch.Tensor | None, new: torch.Tensor | None) -> bool:
    """Check whether a cached tensor holds exactly the values of a new one."""
    if cached is None or new is None:
        return cached is None and new is None
    return (
        cached.shape == new.shape
        and cached.dtype == new.dtype
        and cached.device == new.device
        and torch.equal(cached, new)
    )


class ScatteringTransform(nn.Module):
    """ScatteringTransform base class. Inherits from PyTorch nn.Module.

    This class implements the base logic to compute graph scattering
    transforms with a pooling and an arbitrary wavelet transform
    operators.

    """

    _cached_adj: torch.Tensor | None
    _cached_mask: torch.Tensor | None
    _cached_psi: torch.Tensor | None

    def __init__(
        self,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
        cached: bool = False,
        output: Literal["graph", "node"] = "graph",
        aggregation: Literal["mean", "moments"] = "mean",
        moments: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        """Initialize scattering transform base class.

        This is a base class, and implements only the logic to compute
        an arbitrary scattering transform. The method `get_wavelets`
        must be implemented by the subclass. 

        Scattering coefficients are ordered depth-major, then by parent path, 
        then by scale. 


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
        output: str
            "graph" pools each scattering path over the nodes, giving
            (batch, n_features, n_coefficients), the standard graph scattering transform.
            "node" skips pooling and returns the graph signal 
            (batch, n_features, n_coefficients, n_nodes), used for node-level tasks).
            Defaults to "graph"
        aggregation: str
            How "graph" output pools over nodes. "mean" averages each path, the 
            standard graph scattering transform. 
            "moments" averages `|path| ** q` for each q in `moments`,
            following Gao, Wolf & Hirn 2019, giving n_coefficients *
            len(moments) values per feature, coefficient-major, the so-called 
            geometric scattering transform. Defaults to "mean"
        moments: tuple[int, ...]
            Moment orders used by `aggregation="moments"`. Defaults to (1, 2, 3, 4)
        """
        super().__init__()

        if output not in ("graph", "node"):
            raise ValueError(f"output must be 'graph' or 'node', got {output!r}")
        if aggregation not in ("mean", "moments"):
            raise ValueError(
                f"aggregation must be 'mean' or 'moments', got {aggregation!r}"
            )

        # number of scales
        self.n_scales = n_scales
        # number of layers
        self.n_layers = n_layers

        self.nlin = nlin

        self.cached = cached

        self.output = output
        self.aggregation = aggregation
        self.moments = moments

        # derived from the graph and shaped by its size, so kept out of the
        # state dict; the adjacency is kept to detect when the graph changes
        self.register_buffer("_cached_adj", None, persistent=False)
        self.register_buffer("_cached_mask", None, persistent=False)
        self.register_buffer("_cached_psi", None, persistent=False)

    def path_index(self) -> list[tuple[int, ...]]:
        """List the scale path behind each scattering coefficient.

        Returns
        -------
        list[tuple[int, ...]]
            One tuple of scale indices per coefficient, in output order: `()`
            for the input itself, `(j1,)` for the first layer, `(j1, j2)` for
            the second, and so on
        """
        paths: list[tuple[int, ...]] = [()]
        layer: list[tuple[int, ...]] = [()]
        for _ in range(1, self.n_layers):
            layer = [path + (jj,) for path in layer for jj in range(self.n_scales)]
            paths.extend(layer)
        return paths

    def reset_parameters(self) -> None:
        """Clear the cached filter bank."""
        self._cached_adj = None
        self._cached_mask = None
        self._cached_psi = None

    def _build_wavelets(
        self, W_adj: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Check the adjacency matrix is square, then build its filter bank."""
        assert W_adj.shape[-1] == W_adj.shape[-2], (
            f"W_adj must be square, got shape {tuple(W_adj.shape)}"
        )
        return self.get_wavelets(W_adj, mask)

    def _get_wavelets_cached(
        self, W_adj: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Return the filter bank for `W_adj`, reusing the cache when enabled."""
        if not self.cached or W_adj.requires_grad:
            return self._build_wavelets(W_adj, mask)

        if (
            self._cached_psi is not None
            and _same_tensor(self._cached_adj, W_adj)
            and _same_tensor(self._cached_mask, mask)
        ):
            return self._cached_psi

        psi = self._build_wavelets(W_adj, mask)
        self._cached_adj = W_adj.detach().clone()
        self._cached_mask = None if mask is None else mask.clone()
        self._cached_psi = psi
        return psi

    def get_wavelets(
        self, W_adj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the wavelet operator.

        Subclasses are required to implement this method.

        Parameters
        ----------
        W_adj: torch.Tensor
            Batch of weighted adjacency matrices. Padded nodes have no edges.
        mask: torch.Tensor, optional
            Boolean tensor shaped (batch, n_nodes), true on real nodes. None
            when every graph fills all `n_nodes`.
        """
        raise NotImplementedError

    def get_lowpass(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Compute lowpass filtering/pooling operator.

        This should roughly resemble an average, it alters the output
        scaling factor. For instance averaging with the norm
        of the degree vector scales towards zero, this implementation
        offers a more natural scaling.

        Parameters
        ----------
        mask: torch.Tensor
            Boolean tensor shaped (batch, n_nodes), true on real nodes
        dtype: torch.dtype
            Floating point type of the operator

        Returns
        -------
        lowpass: torch.Tensor
            average pooling operator over each graph's real nodes
        """
        weights = mask.to(dtype)
        lowpass = weights / weights.sum(dim=-1, keepdim=True)

        return lowpass

    def forward(
        self,
        x: torch.Tensor,
        W_adj: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of a generic scattering transform.

        Parameters
        ----------
        x: torch.Tensor
            input batch of graph signals
        W_adj: torch.Tensor
            Batch of weighted adjacency matrices
        mask: torch.Tensor, optional
            Padding mask, boolean tensor shaped (batch, n_nodes)
            Padded entries of `x` and `W_adj` are ignored. Defaults to None

        Returns
        -------
        phi: torch.Tensor
            scattering representation of the input batch

        """
        batch_size, _, n_nodes = x.shape

        if mask is not None:
            assert mask.shape == (batch_size, n_nodes), (
                f"mask must be shaped {(batch_size, n_nodes)}, got {tuple(mask.shape)}"
            )
            assert mask.dtype == torch.bool, f"mask must be bool, got {mask.dtype}"
            # isolate padded nodes and silence their signal, so padding cannot
            # leak into the real nodes whatever the caller filled it with
            W_adj = W_adj * (mask[:, :, None] & mask[:, None, :])
            x = x.masked_fill(~mask[:, None, :], 0)

        psi = self._get_wavelets_cached(W_adj, mask)

        assert psi.shape[0] == batch_size, (
            f"batch size of x ({batch_size}) must match W_adj ({psi.shape[0]})"
        )
        assert psi.shape[-1] == n_nodes, (
            f"node count of x ({n_nodes}) must match W_adj ({psi.shape[-1]})"
        )

        if mask is None:
            mask = torch.ones(batch_size, n_nodes, dtype=torch.bool, device=x.device)

        # apply scattering transform 
        U = self._propagate(x, psi)

        # special case for node outputs, which are not pooled (hence no lowpass operator)
        # directly returns the node 
        if self.output == "node":
            # padding can pick up rounding-level values from the filter bank
            return U.masked_fill(~mask[:, None, None, :], 0)

        lowpass = self.get_lowpass(mask, x.dtype)

        # standard scattering transform, which pools over the nodes with a 
        # lowpass filter
        if self.aggregation == "mean":
            phi: torch.Tensor = einsum(U, lowpass, "b f c n, b n -> b f c")
            return phi

        # moment aggregation, which pools over the nodes with a lowpass filter, then raised to the qth 
        # power for a 'geometric' scattering transform
        moments = torch.stack(
            [
                einsum(U.abs() ** q, lowpass, "b f c n, b n -> b f c")
                for q in self.moments
            ],
            dim=-1,
        )
        return rearrange(moments, "b f c q -> b f (c q)")

    def _propagate(self, x: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Compute every scattering path's graph signal, before any pooling.

        Parameters
        ----------
        x: torch.Tensor
            Batch of graph signals, shaped (batch, n_features, n_nodes)
        psi: torch.Tensor
            Filter bank, shaped (batch, n_scales, n_nodes, n_nodes)

        Returns
        -------
        torch.Tensor
            graph signal shaped (batch, n_features, n_coefficients, n_nodes), in
            `path_index` order
        """
        # the input itself is the depth zero path
        S_x = rearrange(x, "b f n -> b 1 f n")
        signals = [S_x]

        for ll in range(1, self.n_layers):
            S_x_ll = []

            for jj in range(self.n_scales ** (ll - 1)):
                # intermediate repr, one copy per scale to contract against psi
                x_jj = repeat(S_x[:, jj, :, :], "b f n -> b ns f n", ns=self.n_scales)

                # wavelet filtering operation
                psi_x_jj = einsum(x_jj, psi, "b ns f n, b ns n m -> b ns f m")

                # application of non-linearity, yields scattering output
                S_x_ll.append(self.nlin(psi_x_jj))

            # continue iteration through the layer
            S_x = torch.cat(S_x_ll, dim=1)
            signals.append(S_x)

        return rearrange(torch.cat(signals, dim=1), "b c f n -> b f c n")


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
        output: Literal["graph", "node"] = "graph",
        aggregation: Literal["mean", "moments"] = "mean",
        moments: tuple[int, ...] = (1, 2, 3, 4),
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
        output: str
            "graph" or "node", see `ScatteringTransform`. Defaults to "graph"
        aggregation: str
            "mean" or "moments", see `ScatteringTransform`. Defaults to "mean"
        moments: tuple[int, ...]
            Moment orders for `aggregation="moments"`. Defaults to (1, 2, 3, 4)

        """
        super().__init__(n_scales, n_layers, nlin, cached, output, aggregation, moments)

    def get_wavelets(
        self, W_adj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Subclass method used to get wavelet filter bank.

        This method returns diffusion wavelets

        Parameters
        ----------
        W_adj: torch.Tensor
            Batch of weighted adjacency matrices
        mask: torch.Tensor, optional
            Unused because T` is block diagonal

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
        output: Literal["graph", "node"] = "graph",
        aggregation: Literal["mean", "moments"] = "mean",
        moments: tuple[int, ...] = (1, 2, 3, 4),
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
        output: str
            "graph" or "node", see `ScatteringTransform`. Defaults to "graph"
        aggregation: str
            "mean" or "moments", see `ScatteringTransform`. Defaults to "mean"
        moments: tuple[int, ...]
            Moment orders for `aggregation="moments"`. Defaults to (1, 2, 3, 4)

        """
        super().__init__(n_scales, n_layers, nlin, cached, output, aggregation, moments)
        self.use_warp = use_warp

    def warp_func(
        self, spectra: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Compute the spectrum-adaptive warping function.

        Parameters
        ----------
        spectra: torch.Tensor
            Eigenvalues of each graph's Laplacian, sorted ascending, shaped
            (batch, n_nodes)
        mask: torch.Tensor, optional
            Boolean tensor shaped (batch, n_nodes), true on real nodes

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            Per-graph piecewise linear approximation of the spectral CDF
        """
        if mask is None:
            xp, fp = self._warp_points(spectra)
        else:
            # padded nodes add an extra eigenvalue at zero, needs to be removed
            # for CDF to be correct
            n_eigs = spectra.shape[-1]
            points = [
                self._warp_points(spec[n_eigs - int(n_real) :].unsqueeze(0))
                for spec, n_real in zip(spectra, mask.sum(dim=-1), strict=True)
            ]
            width = max(xp_i.shape[-1] for xp_i, _ in points)

            # left-pad by repeating the first point
            xp = torch.cat([_left_pad(xp_i, width) for xp_i, _ in points], dim=0)
            fp = torch.cat([_left_pad(fp_i, width) for _, fp_i in points], dim=0)

        # switched from lambda to partial so the kernel stays picklable
        return partial(_interp, xp=xp.contiguous(), fp=fp.contiguous())

    def _warp_points(self, spectra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the spectral CDF at the points the warp interpolates between."""
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

        return xp, fp

    def get_kernel(
        self, E: torch.Tensor, mask: torch.Tensor | None = None
    ) -> TightHannKernel:
        """Compute TightHann kernel adaptively.

        Parameters
        ----------
        E: torch.Tensor
            Eigenvalues of each graph's Laplacian, shaped (batch, n_nodes)
        mask: torch.Tensor, optional
            Boolean tensor shaped (batch, n_nodes), true on real nodes

        Returns
        -------
        TightHannKernel
            Kernel adapted to each graph's spectrum
        """
        # sort within each graph
        spectra, _ = torch.sort(E, dim=-1)
        return TightHannKernel(
            self.n_scales, spectra[:, -1], self.warp_func(spectra, mask)
        )

    def get_wavelets(
        self, W_adj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Subclass method used to get wavelet filter bank.

        This method returns tight Hann wavelets

        Parameters
        ----------
        W_adj: torch.Tensor
            Batch of weighted adjacency matrices
        mask: torch.Tensor, optional
            Padding mask, boolean tensor shaped (batch, n_nodes)

        Returns
        -------
        psi: torch.Tensor
            tight Hann wavelet operator

        """
        E, V = compute_spectra(W_adj)
        psi = tighthann_wavelets(
            W_adj, self.n_scales, self.get_kernel(E, mask), spectra=(E, V)
        )

        return psi

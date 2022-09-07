"""generic base classes for scattering transform operations

TODO:
    - fix typing for pytorch module
    - confirm symbolic notation
    - add references
    - think more carefully about what should go where wrt torch batching, init
    - add compute moments to base scattering class
    - pass wavelet parameters via kwargs/args
    - add docs
"""
from typing import Any, Callable

# import numpy as np
import torch
from einops import rearrange, reduce, repeat
from scipy.interpolate import interp1d
from torch import nn

from .graph import compute_spectra, normalize_adjacency
from .kernel import TightHannKernel
from .wavelets import diffusion_wavelets, tighthann_wavelets


class ScatteringTransform(nn.Module):  # type: ignore
    """ScatteringTransform base class. Inherits from PyTorch nn.Module

    This class implements the base logic to compute graph scattering
    transforms with an arbitrary pooling and wavelet transform
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
        """Initialize scattering transform base class

        This is a base class, and implements only the logic to compute
        an arbitrary scattering transform. The methods `get_wavelets` and
        `get_pooling` must be implemented by subclasses.

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
        super(ScatteringTransform, self).__init__()

        # adjacency matrix
        self.W_adj = W_adj
        # number of scales
        self.n_scales = n_scales
        # number of layers
        self.n_layers = n_layers

        self.n_nodes = self.W_adj.shape[1]
        assert self.W_adj.shape[1] == self.W_adj.shape[2]

        self.nlin = nlin

    def get_wavelets(self) -> torch.Tensor:
        """Compute wavelet operator. Subclasses are required to
        implement this method"""

        raise NotImplementedError

    def get_lowpass(self) -> torch.Tensor:
        """Compute pooling operator. Subclasses are required to implement this method"""

        raise NotImplementedError

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

        lowpass = self.get_lowpass()
        psi = self.get_wavelets()

        # compute first scattering layer, low pass filter via matmul
        # p = 1, padding dimension
        phi = torch.einsum("b f n, b n p -> b f p", x, lowpass)

        # reshape inputs for loop
        S_x = rearrange(x, "b f n -> b 1 f n")
        lowpass = rearrange(lowpass, "b n 1 -> b 1 n 1")
        lowpass = repeat(lowpass, "b 1 n 1 -> b (1 ns) n 1", ns=self.n_scales)

        for ll in range(1, self.n_layers):

            S_x_ll = torch.empty([batch_size, 0, n_features, self.n_nodes])

            for jj in range(self.n_scales ** (ll - 1)):

                # intermediate repr
                x_jj = rearrange(S_x[:, jj, :, :], "b f n -> b 1 f n")

                # wavelet filtering operation, matrix multiply
                psi_x_jj = torch.einsum("b p f n, b l n n -> b l f n", x_jj, psi)

                # application of non-linearity, yields scattering output
                S_x_jj = self.nlin(psi_x_jj)

                # concat scattering scale for the layer
                S_x_ll = torch.cat((S_x_ll, S_x_jj), axis=1)

                # compute scattering representation, matrix multiply
                phi_jj = torch.einsum("b l f n, b l n p -> b l f p", S_x_jj, lowpass)
                phi_jj = rearrange(phi_jj, "b l f 1 -> b f l")

                phi = torch.cat((phi, phi_jj), axis=2)

            S_x = S_x_ll.clone()  # continue iteration through the layer

        return phi


class Diffusion(ScatteringTransform):
    """Diffusion scattering transform.

    Subclass of `ScatteringTransform`, implements `get_wavelets` and `get_lowpass`
    methods.

    """

    def __init__(
        self,
        W_adj: torch.Tensor,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
    ) -> None:
        """Initialize diffusion scattering transform

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
        """Subclass method used to get wavelet filter bank

        This method returns diffusion wavelets

        Returns
        -------
        psi: torch.Tensor
            diffusion wavelet operator

        """

        W_norm = normalize_adjacency(self.W_adj)

        # compute diffusion matrix
        T = 1 / 2 * (torch.eye(self.n_nodes) + W_norm)
        # compute wavelet operator
        psi = diffusion_wavelets(T, self.n_scales)

        return psi

    def get_lowpass(self) -> torch.Tensor:
        """subclass method used to get lowpass pooling operator

        Returns
        -------
        lowpass: torch.Tensor
            lowpass pooling operator

        """

        # compute degree vector. ni and nj are equivalent, used to disambiguate
        d = reduce(self.W_adj, "b ni nj -> b ni", "sum")
        # normalize
        lowpass = d / torch.norm(d, 1)
        lowpass = rearrange(lowpass, "b ni -> b ni 1")

        return lowpass


class TightHann(ScatteringTransform):
    """TightHann scattering transform.

    Subclass of `ScatteringTransform`, implements `get_wavelets` and `get_lowpass`
    methods.

    """

    def __init__(
        self,
        W_adj: torch.Tensor,
        n_scales: int,
        n_layers: int,
        nlin: Callable[[torch.Tensor], torch.Tensor] = torch.abs,
    ) -> None:
        """Initialize diffusion scattering transform

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

        self.warp_func = self.warp_func()  # type: ignore

    def warp_func(self, use: bool = True) -> torch.Tensor:
        """Implements spectrum-adaptive warping function"""

        E, V = compute_spectra(self.W_adj)
        self.spectra, _ = torch.sort(E.reshape(-1))  # change this
        self.max_eig = self.spectra.max()

        cdf_ = torch.arange(0, len(self.spectra)) / (len(self.spectra) - 1.0)
        intvl = int(len(self.spectra) / 5 - 1)

        if use:
            return interp1d(
                self.spectra[0::intvl], cdf_[0::intvl], fill_value="extrapolate"
            )
        else:
            return interp1d(self.spectrum, cdf_, fill_value="extrapolate")

    def get_kernel(self) -> TightHannKernel:
        """compute TightHann kernel adaptively"""

        omega = lambda eig: torch.tensor(self.warp_func(eig.numpy()))

        return TightHannKernel(self.n_scales, self.max_eig, omega)

    def get_wavelets(self) -> torch.Tensor:
        """Subclass method used to get wavelet filter bank

        This method returns diffusion wavelets

        Returns
        -------
        psi: torch.Tensor
            diffusion wavelet operator

        """

        # compute wavelet operator
        psi = tighthann_wavelets(self.W_adj, self.n_scales, self.get_kernel())

        return psi

    def get_lowpass(self) -> torch.Tensor:
        """subclass method used to get lowpass pooling operator

        TODO: check if this can be added to the base class, might not need this...

        Returns
        -------
        lowpass: torch.Tensor
            lowpass pooling operator

        """

        mu = (1 / self.W_adj.shape[1]) * torch.ones(
            self.W_adj.shape[0], self.W_adj.shape[1]
        )  # , device=self.device)
        mu = rearrange(mu, "b ni -> b ni 1")  # added this ...

        return mu

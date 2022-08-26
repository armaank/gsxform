"""generic base classes for scattering transform operations

TODO:
    - this may not even need to be a nn.Module, no params
    - fix typing for pytorch module
    - confirm symbolic notation
    - add references
    - think more carefully about what should go where wrt torch batching, init
    - add compute moments to base scattering class
    - pass wavelet parameters via kwargs/args
    - add docs
"""
from typing import Any

import numpy as np
import torch
from torch import nn

from .graph import compute_spectra, normalize_adjacency
from .wavelets import diffusion_wavelets, hann_wavelets, spline_wavelets


class ScatteringTransform(nn.Module):  # type: ignore
    """ScatteringTransform base class. Inherets from PyTorch nn.Module"""

    def __init__(self, W_adj: torch.Tensor, J: int, L: int, **kwargs: Any) -> None:
        """Initilize scattering transform base class

        This is a base class, and implements only the logic to compute
        an arbitrary scattering transform. the methods `get_wavelets` and
        `get_pooling` must be implemented by subclasses.

        Parameters
        ----------
        W_adj: torch.Tensor
            Weighted adjacency matrix
        J: int
            Number of scales to use in wavelet transform
        L: int
            Bumber of layers in the scattering transform
        **kwargs: Any
            Additional keyword arguments
        """
        super(ScatteringTransform, self).__init__()

        # adjacency matrix
        self.W_adj = W_adj
        # number of scales
        self.J = J
        # number of layers
        self.L = L

        # TODO: check this, might be batched
        self.n_nodes = self.W_adj.shape[1]
        assert self.W_adj.shape[1] == self.W_adj.shape[2]

        # placeholders for wavelet and pooling operator
        self.lowpass: torch.Tensor = None
        self.psi: torch.Tensor = None

        # non linearity
        self.nlin = torch.abs

    def extra_repr(self) -> str:
        return f"gsxform(N={self.n_nodes}, J={self.J}, L={self.L}"

    def __str__(self) -> str:
        return f"Graph scattering transform: {self.n_nodes} nodes, {self.J} scales,\
                {self.L} layers"

    def get_wavelets(self) -> None:
        """Compute wavelet filterbank. Subclasses are required to
        implement this method"""

        raise NotImplementedError

    def get_pooling(self) -> None:
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

        # if len(x) == 2:
        # assume single graph input, not batched...
        #    batch_size = 1
        #    x = x.transpose() # x is transposed...
        # else:
        batch_size = x.shape[0]

        n_features = x.shape[1]

        lowpass = self.lowpass.reshape([1, self.n_nodes, 1])
        print(x.shape)
        print(lowpass.shape)
        # compute first scattering layer, low pass filter input
        phi = torch.matmul(x, lowpass)

        # reshape inputs for loop
        S_x = x.reshape(batch_size, 1, n_features, self.n_nodes)
        lowpass = lowpass.reshape(1, 1, self.n_nodes, 1)
        lowpass = torch.tile(
            lowpass,
            [
                1,
                self.J,
                1,
                1,
            ],
        )
        psi = self.psi.reshape(1, self.J, self.n_nodes, self.n_nodes)

        for ll in range(1, self.L):
            S_x_ll = torch.empty([batch_size, 0, n_features, self.n_nodes])
            # layer_output = torch.empty([batch_size, 0, n_features, self.N])
            for jj in range(self.J ** (ll - 1)):

                x_jj = S_x[:, jj, :, :]  # intermediate repr.
                x_jj = x_jj.reshape(batch_size, 1, n_features, self.n_nodes)
                psi_x_jj = torch.matmul(x_jj, psi)  # wavelet filtering operation
                S_x_jj = self.nlin(psi_x_jj)  # scattering output
                S_x_ll = torch.cat(
                    (S_x_ll, S_x_jj), axis=1
                )  # concat scattering scale for the layer

                # compute scattering representation
                phi_jj = torch.matmul(S_x_jj, lowpass)
                # store coefficients
                phi_jj = phi_jj.squeeze(3)
                phi_jj = phi_jj.transpose(0, 2, 1)
                phi = torch.cat((phi, phi_jj), axis=2)

            S_x = S_x_ll.copy()  # continue iteration through the layer

        return phi


class Diffusion(ScatteringTransform):
    """Diffusion scattering transform."""

    def __init__(self, W_adj: torch.Tensor, J: int, L: int, **kwargs: Any) -> None:
        """Initilize diffusion scattering transform

        Parameters
        ----------

        """
        # super().__init__(Diffusion, self)  # W_adj, J, L)
        super().__init__(W_adj, J, L)
        # self.n_nodes = self.W_adj.shape[1]
        self.psi = self.get_wavelets()
        self.lowpass = self.get_lowpass()

        pass

    def get_wavelets(self) -> torch.Tensor:
        """subclass method used to get wavelet filter bank


        Returns
        -------

        """

        W_norm = normalize_adjacency(self.W_adj)

        # compute diffusion matrix
        T = 1 / 2 * (torch.eye(self.n_nodes) + W_norm)
        # compute wavelet operator
        psi = diffusion_wavelets(T, self.J)

        return psi

    def get_lowpass(self) -> torch.Tensor:
        """subclass method used to get lowpass pooling operator

        Returns
        -------

        """

        # compute lowpass operator
        d = self.W_adj.sum(1)
        lowpass = d / torch.norm(d, 1)

        return lowpass


class Spline(ScatteringTransform):
    """Spline, monic polynomial scattering transform class"""

    def __init__(self, W_adj: torch.Tensor, J: int, L: int, **kwargs: Any) -> None:
        """Initilize spline scattering transform

        Parameters
        ----------

        """
        # super().__init__(W_adj: torch.Tensor, J: int, L:int)
        self.alpha = 2
        self.beta = 2
        self.K = 2
        self.psi = self.get_wavelets()  # alpha=2, beta=2, K=2)
        self.lowpass = self.get_lowpass()

        pass

    def get_wavelets(self) -> torch.Tensor:
        """subclass method used to get wavelet filter bank.

        Returns
        -------


        """

        # compute gft
        E, V = compute_spectra(self.W_adj)
        eig_max = torch.max(torch.diag(E))

        x1 = torch.diag(E)[np.floor(self.n_nodes / 4)]
        x2 = torch.diag(E)[np.ciel(3 * self.n_nodes / 4)]

        # compute wavelet operator
        psi = spline_wavelets(
            V, E, self.J, self.alpha, self.beta, x1, x2, self.K, eig_max
        )

        return psi

    def get_lowpass(self) -> torch.Tensor:
        """subclass method used to get lowpass pooling operator.

        Returns
        -------


        """

        # compute lowpass operator
        lowpass = 1 / self.n_nodes * torch.ones(self.n_nodes)

        return lowpass


class TightHann(ScatteringTransform):
    """Tight Hann scattering transform class"""

    def __init__(self, W_adj: torch.Tensor, J: int, L: int, **kwargs: Any) -> None:
        """Initilize tight-hann scattering transform

        Parameters
        ----------

        """

        # super().__init__(W_adj: torch.Tensor, J: int, L:int)
        self.R = 3
        self.warp = None
        self.psi = self.get_wavelets()
        self.lowpass = self.get_lowpass()

        pass

    def get_wavelets(self) -> torch.Tensor:
        """subclass method used to get wavelet filter bank


        Returns
        -------

        """

        # compute gft
        E, V = compute_spectra(self.W_adj)
        eig_max = torch.max(torch.diag(E))

        # compute wavelet operator
        psi = hann_wavelets(V, E, self.J, self.R, eig_max, self.warp)

        return psi

    def get_lowpass(self) -> torch.Tensor:
        """subclass method used to get lowpass pooling operator


        Returns
        -------


        """

        # compute lowpass operator
        lowpass = 1 / self.n_nodes * torch.ones(self.n_nodes)

        return lowpass

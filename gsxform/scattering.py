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
from typing import Any

import torch
from torch import nn

# from .graph import compute_spectra, normalize_adjacency
from .wavelets import diffusion_wavelets


class ScatteringTransform(nn.Module):  # type: ignore
    """ScatteringTransform base class. Inherets from PyTorch nn.Module"""

    def __init__(
        self, W_adj: torch.Tensor, n_scales: int, n_layers: int, **kwargs: Any
    ) -> None:
        """Initilize scattering transform base class

        This is a base class, and implements only the logic to compute
        an arbitrary scattering transform. the methods `get_wavelets` and
        `get_pooling` must be implemented by subclasses.

        Parameters
        ----------
        W_adj: torch.Tensor
            Weighted adjacency matrix
        n_scales: int
            Number of scales to use in wavelet transform
        n_layers: int
            Number of layers in the scattering transform
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

        # TODO: check this, might be batched
        self.n_nodes = self.W_adj.shape[1]
        assert self.W_adj.shape[1] == self.W_adj.shape[2]

        # placeholders for wavelet and pooling operator
        # self.lowpass: torch.Tensor = None
        # self.psi: torch.Tensor = None

        # non linearity, make this an arg
        self.nlin = torch.abs

    def extra_repr(self) -> str:
        return f"gsxform(N={self.n_nodes}, J={self.n_scales}, L={self.n_layers}"

    def __str__(self) -> str:
        return f"Graph scattering transform: {self.n_nodes} nodes, {self.n_scales} scales,\
                {self.n_layers} layers"

    def get_wavelets(self) -> torch.Tensor:
        """Compute wavelet filterbank. Subclasses are required to
        implement this method"""

        raise NotImplementedError

    def get_pooling(self) -> torch.Tensor:
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

        # lowpass = self.lowpass.reshape([batch_size, self.n_nodes, 1])
        lowpass = self.get_lowpass()
        psi = self.get_wavelets()

        # compute first scattering layer, low pass filter input
        phi = torch.matmul(x, lowpass.unsqueeze(2))

        # reshape inputs for loop
        # S_x = x.reshape(batch_size, 1, n_features, self.n_nodes)
        # lowpass = lowpass.reshape(batch_size, 1, self.n_nodes, 1)
        # lowpass = torch.tile(
        #    lowpass,
        #    [
        #        1,
        #        self.J,
        #        1,
        #        1,
        #    ],
        # )
        # psi = self.psi.reshape(batch_size, self.J, self.n_nodes, self.n_nodes)
        S_x = x.unsqueeze(1)
        lowpass = lowpass.unsqueeze(1).unsqueeze(3).repeat(1, self.n_scales, 1, 1)

        for ll in range(1, self.n_layers):
            S_x_ll = torch.empty([batch_size, 0, n_features, self.n_nodes])
            # layer_output = torch.empty([batch_size, 0, n_features, self.N])
            for jj in range(self.n_scales ** (ll - 1)):

                x_jj = S_x[:, jj, :, :].unsqueeze(1)  # intermediate repr.
                # x_jj = x_jj.reshape(batch_size, 1, n_features, self.n_nodes)
                psi_x_jj = torch.matmul(x_jj, psi)  # wavelet filtering operation

                S_x_jj = self.nlin(psi_x_jj)  # scattering output
                S_x_ll = torch.cat(
                    (S_x_ll, S_x_jj), axis=1
                )  # concat scattering scale for the layer

                # compute scattering representation
                phi_jj = torch.transpose(
                    (torch.matmul(S_x_jj, lowpass).squeeze(3)), 2, 1
                )
                # store coefficients
                # phi_jj = phi_jj.squeeze(3)
                # phi_jj = phi_jj.permute(0, 2, 1)
                phi = torch.cat((phi, phi_jj), axis=2)

            S_x = S_x_ll.clone()  # continue iteration through the layer

        return phi


class Diffusion(ScatteringTransform):
    """Diffusion scattering transform."""

    def __init__(
        self, W_adj: torch.Tensor, n_scales: int, n_layers: int, **kwargs: Any
    ) -> None:
        """Initilize diffusion scattering transform

        Parameters
        ----------

        """
        super().__init__(W_adj, n_scales, n_layers)

        # self.psi = self.get_wavelets()
        # self.lowpass = self.get_lowpass()

        pass

    def get_wavelets(self) -> torch.Tensor:
        """subclass method used to get wavelet filter bank


        Returns
        -------

        """

        # W_norm = normalize_adjacency(self.W_adj)

        # compute diffusion matrix
        # T = 1 / 2 * (torch.eye(self.n_nodes) + W_norm)
        # compute wavelet operator
        psi = diffusion_wavelets(self.W_adj, self.n_scales)

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

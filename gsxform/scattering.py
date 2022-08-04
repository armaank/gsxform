"""generic base classes for scattering transform operations

TODO:
    - write scattering base class
    - this may not even need to be a nn.Module, no params
    - fix typing for pytorch module
    - confirm symbolic notation
"""
from typing import Any

import torch
from torch import nn

from .graph import normalize_adjacency
from .wavelets import diffusion_wavelets


class ScatteringTransform(nn.Module):  # type: ignore
    def __init__(self, W_adj: torch.Tensor, J: int, L: int, **kwargs: Any) -> None:
        """Initilize ScatteringTransform method"""
        super(ScatteringTransform, self).__init__()

        # adjacency matrix
        self.W_adj = W_adj
        # number of scales
        self.J = J
        # number of layers
        self.L = L

        # TODO: check this, might be batched
        self.n_nodes = self.W_adj.shape[0]

        # placeholders for wavelet and pooling operator
        self.lowpass: torch.Tensor = None
        self.psi: torch.Tensor = None

        # non linearity
        self.nlin = torch.abs

    def extra_repr(self) -> str:
        return f"gsxform(N={self.N}, J={self.J}, L={self.L}"

    def __str__(self) -> str:
        return f"Graph scattering transform: {self.N} nodes, {self.J} scales,\
                {self.L} layers"

    def get_wavelets(self) -> None:
        """compute wavelet filterbank"""

        raise NotImplementedError

        pass

    def get_pooling(self) -> None:
        """compute pooling operator"""

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform method, the forward pass."""

        batch_size = x.shape[0]
        n_features = x.shape[1]

        # compute first scattering layer, low pass filter input
        phi = torch.matmul(x, self.lowpass)

        # reshape inputs for loop
        S_x = x.reshape(batch_size, 1, n_features, self.n_nodes)
        lowpass = self.lowpass.reshape(1, 1, self.n_nodes, 1)
        lowpass = torch.tile(
            lowpass,
            [
                1,
                self.n_nodes,
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
                    (S_x_ll, S_x_jj)
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
    """Diffusion scattering transform class"""

    def __init__(self, W_adj: torch.Tensor, J: int, L: int, **kwargs: Any) -> None:
        # super().__init__(W_adj: torch.Tensor, J: int, L:int)

        self.psi = self.get_wavelets()
        self.lowpass = self.get_lowpass()

        pass

    def get_wavelets(self) -> torch.Tensor:
        """subclass method used to get wavelet filter bank"""

        W_norm = normalize_adjacency(self.W_adj)

        # compute diffusion matrix
        T = 1 / 2 * (torch.eye(self.num_nodes) + W_norm)
        # compute wavelet operator
        psi = diffusion_wavelets(T, self.J)

        return psi

    def get_lowpass(self) -> torch.Tensor:
        """subclass method used to get lowpass pooling operator"""

        # compute lowpass operator
        d = self.W_adj.sum(1)
        lowpass = d / torch.norm(d, 1)

        return lowpass


# class Spline(ScatteringTransform):
#    """Cubic spline scattering transform class
#
#    """


#   def __init__(self, ):

#        pass


# class Hann(ScatteringTransform):
#    """Tight Hann scattering transform class
#    """


#    def __init__(self, ):


#        pass


# class DiffusionScattering(nn.Module):
#    def __init__(self, W_adj:torch.Tensor, J:int, L:int):
#        super(DiffusionScattering, self).__init__()
#
#        # adjacency matrix
#        self.W_adj = W_adj
#        # number of scales
#        self.J = J
#        # number of layers
#        self.L = L
#
#        # TODO: check this, might be batched
#        self.N = self.W_adj.shape[0]
#
#        # degree vector
#        self.d = W_adj.sum(1)
#
#        pass
#
#    def get_wavelets(self) -> None:
#
#        W_norm = graph.normalize_adjacency(self.W_adj)
#
#        # diffusion operator
#        self.T = 1/2 * (torch.eye(self.N) + W_norm)
#
#        # pooling operator
#
#        self.U = d / torch.linalg.norm(d, 1)
#
#        self.psi = wavelets.diffusion_wavelets(self.T, self.J)
#
#        pass
#
#    def forward(self, x:torch.Tensor) -> torch.Transform:
#
#        return x
#
#
# class SplineScattering(nn.Module):
#    def __init__(self, W_adj:torch.Tensor, J:int, L:int, alpha, beta, k, norm):
#        super(SplineScattering, self).__init__()
#
#        self.W_adj = W_adj
#        self.J = J
#        self.L = L
#
#        self.N = self.W_adj.shape[0]
#
#        pass
#

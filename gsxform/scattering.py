"""generic base classes for scattering transform operations

TODO:
    - write scattering base class
    - this may not even need to be a nn.Module, no params
    - fix typing for pytorch module
"""
from typing import Any

import torch
from torch import nn

# import graphs
# import wavelets


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
        self.N = self.W_adj.shape[0]

        # placeholders for wavelet and pooling operator
        # TODO: figure out better name than U
        # self.U = None
        # self.psi = None
        # non linearity
        # self.nlin = torch.abs

    def extra_repr(self) -> str:
        return f"gsxform(N={self.N}, J={self.J}, L={self.L}"

    def __str__(self) -> str:
        return f"Graph scattering transform: {self.N} nodes, {self.J} scales,\
                {self.L} layers"

    def _get_wavelets(self) -> None:
        """compute wavelet filterbank"""

        raise NotImplementedError

        pass

    def _get_pooling(self) -> None:
        """compute pooling operator"""

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform method, the forward pass."""

        batch_size = x.shape[0]
        n_features = x.shape[1]

        # phi = torch.matmul(x, self.U)

        # reshape inputs for loop
        x = x.reshape(batch_size, 1, n_features, self.N)
        # self.U = self.U.reshape(1, 1, self.N, 1)

        # stub, re-write this
        for ll in range(1, self.L):

            for jj in range(self.J ** (ll - 1)):

                pass
            pass

        return x


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

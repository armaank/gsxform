"""Implementations of kernel functions used to build graph wavelets

TODO:
    - add references
    - rework into a function, no need for a kernel class
"""

from typing import Callable, Union

import numpy as np
import torch


class TightHannKernel(object):
    """TightHannKernel class.

    Thie class constructs a spectrum-adaptive tight-hann kernel function used
    in its corresponding wavelet transform. Based off of the implementation
    from Tabar et. al 2021 of the algorithm originally described in
    Shuman et. al 2015

    """

    def __init__(
        self,
        n_scales: int,
        max_eig: torch.Tensor,
        omega: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
    ) -> None:
        """Initialize TightHannKernel class

        Parameters
        ----------
        n_scales: int
            number of scales used in wavelet transform
        max_eig: torch.Tensor:
            the maximum eigenvalue of the graph laplacian. Used for scaling purposes
        omega: Union[Callable[[torch.Tensor], torch.Tensor], None]
            warping function. Defaults to None

        """

        self.n_scales = n_scales
        self.K = 1
        self.R = 3.0
        self.max_eig = max_eig

        if omega is not None:
            self.omega = omega
            self.max_eig = self.omega(self.max_eig.float())

        # dilation factor, might need to reverse this to account for swapped bounds...
        # self.d = (self.M + 1 - self.R) / (self.R * self.max_eig)
        self.d = self.R * self.max_eig / (self.n_scales + 1 - self.R)
        # hann kernel functional form
        self.kernel: Callable[[torch.Tensor], torch.Tensor] = (
            lambda eig: sum(
                [
                    0.5 * torch.cos(2 * np.pi * (eig / self.d - 0.5) * k)
                    for k in range(self.K + 1)
                ]
            )
            * (eig >= 0)
            * (eig <= self.d)
        )

    def get_adapted_kernel(self, eig: torch.Tensor, scale: int) -> torch.Tensor:
        """compute spectrum adapted kernels.
        return self.kernel(self.omega(eig) - self.d / self.R * (scale - self.R + 1))

        Parameters
        ----------
        eig: torch.Tensor
            input tensor of eigenvalues of the graph laplacian
        scale: int
            The scale parameter of the specific kernel. Not to be confused
            with `n_scales`, which is the total number of scales used
            by the wavelet transform.

        Returns
        --------
        adapted_kernel: torch.Tensor
            scale-specific adapted kernel

        """
        adapted_kernel = self.kernel(
            self.omega(eig) - self.d / self.R * (scale - self.R + 1)
        )
        return adapted_kernel

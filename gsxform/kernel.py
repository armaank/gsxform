"""Implementations of kernel functions used to build graph wavelets

TODO:
    - add references
    - rework into a function, no need for a kernel class
"""

from typing import Any, Callable, Union

import numpy as np
import torch


class TightHannKernel:
    def __init__(
        self,
        M: int,
        max_eig: torch.Tensor,
        omega: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
    ):

        self.M = M  # number of scales
        self.K = 1
        self.R = 3.0
        self.max_eig = max_eig

        if omega is not None:
            self.omega = omega
            self.max_eig = self.omega(self.max_eig.float())

        # dilation factor, might need to reverse this to account for swapped bounds...
        # self.d = (self.M + 1 - self.R) / (self.R * self.max_eig)
        self.d = self.R * self.max_eig / (self.M + 1 - self.R)
        # hann kernel
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

    def adapted_kernels(
        self, eig: float, m: int
    ) -> Any:  # Callable[[torch.Tensor], torch.Tensor]:
        """compute spectrum adapted kernels. check return type"""
        return self.kernel(self.omega(eig) - self.d / self.R * (m - self.R + 1))

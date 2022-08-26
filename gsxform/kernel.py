"""Implementations of kernel functions used to build graph wavelets

TODO:
    - add references
"""

from typing import Callable, Union

import numpy as np
import torch


def hann_kernel(
    x: torch.Tensor,
    J: int,
    R: float,
    gamma: float,
    K: float = 1,
    omega: Union[Callable[[torch.Tensor], torch.Tensor], None] = None,
) -> torch.Tensor:
    """Evaluate the Hann kernel function. Equivalent to the half-cosine
    kernel.

    Parameters
    ----------
    x: torch.Tensor
        input tensor
    J: int
        Number of filters (scales). Equivalent to M in Eq. 9 from Shuman et. al
    R: float
        Scaling factor in Eq. 9 from Shuman et. al. 2 < R < M
    gamma: float
        maximum eigenvalue
    K: float, default=1
        Scaling factor in Eq. 9 from Shuman et. al. K < R/2
    omega: Union[Callable, None], default=None
        Optional warping function


    Returns
    -------
    torch.Tensor
        the values of the parameterized Hann kernel for values of x

    """

    # number of filters (scales). folowing convention of
    # Shuman et, al
    M = J
    assert 2 < R < M

    # if present, apply scaling function to max eigenvalue
    if omega is not None:
        gamma = omega(gamma)

    # dilation factor
    d = (M + 1 - R) / (R * gamma)
    # Hann Kernel
    g = 0.5 + 0.5 * torch.cos(2 * np.pi * d * x * K + 0.5)
    g[x >= 0] = 0
    g[x < d] = 0

    return g


def spline_kernel(
    x: torch.Tensor, alpha: int = 2, beta: int = 2, x1: int = 1, x2: int = 2
) -> torch.Tensor:
    """Evaluate a cubic spline kernel (monic polynomials).

    Parameters
    ----------
    x: torch.Tensor
        input tensor
    alpha: int, default = 2
        kernel parameter.
    beta: int, default = 2
        kernel parameter. Setting alpha=beta is convention
    x1: int, defualt = 1
        kernel parameter.
    x2: int, default = 2
        kernel parameter.

    Returns
    -------
    torch.Tensor
        the values of the parameterized spline kernel for values of x
    """
    print(x)
    print(x.shape)
    x = x[0]
    # alternatively, x.flatten?
    # print(x.shape)
    coeffs = torch.Tensor(
        [
            [1, x1, x1**2, x1**3],
            [1, x2, x2**2, x2**3],
            [0, 1, 2 * x1, 3 * x1**2],
            [0, 1, 2 * x2, 3 * x2**2],
        ]
    )

    # continuity constraints for the cubic polynomial
    # alpha=beta=2, x1=1, x2=2
    constraints = torch.Tensor([1, 1, alpha / x1, -beta / x2])

    # solving for polynomial coeffs
    s = torch.linalg.solve(coeffs, constraints)

    # defining monic spline peacemeal according to eq. 65 of
    # Hammond et. al along three boundaries
    b1 = x < x1
    b2 = (x >= x1) * (x < x2)
    b3 = x >= x2
    print(x)
    print(x.shape)

    # shape changed to 1 to ignore batch size?
    g = np.zeros(x.shape[0])
    print(g.shape)
    print(alpha)

    g[b1] = (x1 ** (-alpha)) * (x[b1] ** alpha)
    g[b2] = s[1] + 2 * s[2] * x[b2] + 3 * s[3] * x[b2] ** 2
    g[b3] = x2 ** (beta) * (x[b3] ** (-beta))

    return g

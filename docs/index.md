# gsxform

 Wavelet scattering transforms on graphs via PyTorch

[![tests](https://github.com/armaank/gsxform/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/armaank/gsxform/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/armaank/gsxform/branch/main/graph/badge.svg?token=AUFSGAPB4O)](https://codecov.io/gh/armaank/gsxform)
[![docs](https://github.com/armaank/gsxform/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/armaank/gsxform/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/gsxform)](https://pypi.org/project/gsxform/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![license](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/513695351.svg)](https://zenodo.org/badge/latestdoi/513695351)
 ---

`gsxform` is a package for constructing graph scattering transforms, leveraging PyTorch
to allow for GPU based computation.
Using PyTorch, `gsxform` offers the ability to more
easily build models that use both scattering transform and neural network components.

`gsxform` is first and foremost a research project and is being continuously refined.
Behavior can potentially be unstable and consistency is not guaranteed.

## Installation

`gsxform` requires Python 3.10 or later.

### Latest version (recommended)


```bash
uv add "gsxform @ git+https://github.com/armaank/gsxform"
```

Or, to use a specific branch not yet merged to main

```bash
uv add "gsxform @ git+https://github.com/armaank/gsxform@branch-name"
```

### Stable Version

To install the latest stable release from PyPI:

```bash
uv add gsxform
# or
pip install gsxform
```

### Development version

`gsxform` uses [`uv`](https://docs.astral.sh/uv/) to manage environments and
dependencies. To work on `gsxform` itself, clone the repository and run `make install`:

```bash
git clone https://github.com/armaank/gsxform.git
cd gsxform
make install
```

This creates the virtual environment, installs `gsxform` with development and
documentation dependencies, and sets up the `pre-commit` hooks. The
[development guide](https://armaank.github.io/gsxform/development/) covers the `make`
targets, code style, branching, and the release process.


## License

The original code of this repository is released under the
[BSD 3.0-Clause Licence](https://github.com/armaank/gsxform/blob/main/LICENSE).
Modifications, adaptations and derivative work is encouraged!

## Citation

If you use `gsxform`, please cite using the
[Zenodo DOI](https://doi.org/10.5281/zenodo.7069113)

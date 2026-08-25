# gsxform

 Wavelet scattering transforms on graphs via PyTorch

[![tests](https://github.com/armaank/gsxform/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/armaank/gsxform/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/armaank/gsxform/branch/main/graph/badge.svg?token=AUFSGAPB4O)](https://codecov.io/gh/armaank/gsxform)
[![docs](https://github.com/armaank/gsxform/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/armaank/gsxform/actions/workflows/docs.yml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
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

### Official Release

`gsxform` is available on PyPi:

```bash
pip install gsxform
```

`gsxform` supports Python 3.10 and newer.

### Pre-releases

The most up-to-date version of `gsxform` can be installed via git:

```bash
pip install git+https://github.com/armaank/gsxform.git
```

### Development version

`gsxform` uses [`uv`](https://docs.astral.sh/uv/) to manage environments and
dependencies. To work on `gsxform` itself, clone the repository and run `make install`,
which creates the virtual environment, installs `gsxform` in editable mode along with
the development and documentation dependencies, and sets up the `pre-commit` hooks:

```bash
git clone https://github.com/armaank/gsxform.git
cd gsxform
make install
```

From there, `make` on its own lists the available targets:

```bash
make tests      # run the test suite with coverage
make lint       # check formatting and lint rules
make typecheck  # run mypy
make docs       # build the documentation site
```

See the [installation page](install.md) for more detail.

## License

The original code of this repository is released under the
[BSD 3.0-Clause Licence](https://github.com/armaank/gsxform/blob/main/LICENSE).
Modifications, adaptations and derivative work is encouraged!

## Citation

If you use `gsxform`, please cite using the [Zenodo DOI](https://zenodo.org/record/7069114)

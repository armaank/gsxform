# gsxform
 Wavelet scattering transforms on graphs via PyTorch
 
[![tests](https://github.com/armaank/gsxform/workflows/tests/badge.svg)](https://github.com/armaank/gsxform/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/armaank/gsxform/branch/main/graph/badge.svg?token=AUFSGAPB4O)](https://codecov.io/gh/armaank/gsxform)
[![docs](https://github.com/armaank/gsxform/workflows/docs/badge.svg)](https://github.com/armaank/gsxform/actions/workflows/docs.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
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

## TODO

* clean up readme, add additional formatting
* check dependencies 

## Installation

### Official Release

`gsxform` is available on PyPi:
```bash
pip install gsxform
```

### Pre-releases

The most up-to-date version of `gsxform` can be installed via git:
```bash
pip install git+https://github.com/armaank/gsxform.git
```

## License 
The original code of this repository is released under the
[BSD 3.0-Clause Licence](https://github.com/armaank/gsxform/blob/main/LICENSE).
Modifications, adaptations and derivative work is encouraged!

## Citation 
If you use `gsxform`, please cite using the [Zenodo DOI](https://zenodo.org/record/7069114)


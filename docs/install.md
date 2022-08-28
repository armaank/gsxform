# Installation

`gsxform` is written without complex dependencies, it can be installed using `pip` or
from source. Currently, `gsxform` has only been tested with Python3.9, previous versions
of Python are not explicitly supported.

### Using `pip`

`gsxform` can be installed using `pip`

```bash
pip install gsxform
```

### From source

The code for `gsxform` can be downloaded and installed as follows:
```bash
git clone https://github.com/armaank/gsxform.git
cd gsxform
python setup.py install
```

## Development

To contribute to `gxform`, download the source code, setup a conda environment and
`source` the setup script to install all of the `pre-commit` hooks to ensure
appropriate typing and code formatting. 
```bash
git clone https://github.com/armaank/gsxform.git
cd gsxform
make conda
source scripts/setup.sh
```

Please follow the [NumPy development
workflow](https://numpy.org/doc/1.14/dev/gitwash/development_workflow.html) naming
convention for pull requests. The test suite is run and the documentation site is
published automatically on every push to the `main` branch via Github Actions

### Testing

To run the unit tests locally using `pytest`, from the root project directory execute
```bash
make tests
```

### Documentation


To preview documentation locally, from the root project directory execute:
```bash
make docs
```



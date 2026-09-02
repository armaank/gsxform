# Installation

`gsxform` is written without complex dependencies, it can be installed using `pip` or
from source. `gsxform` supports Python 3.10 and newer.

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
pip install .
```

## Development

To work on `gsxform` itself, clone the source and run `make install`, which creates the
development environment and installs the `pre-commit` hooks:

```bash
git clone https://github.com/armaank/gsxform.git
cd gsxform
make install
```

See the [development guide](development.md) for the `make` targets, code style,
branching model, and release process.

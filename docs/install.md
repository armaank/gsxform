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

`gsxform` uses [`uv`](https://docs.astral.sh/uv/) to manage environments and
dependencies. To contribute, download the source code and run `make install` to
create the development environment and install the `pre-commit` hooks that enforce
typing and code formatting.
```bash
git clone https://github.com/armaank/gsxform.git
cd gsxform
make install
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

### Linting and type checking

Formatting and lint rules are enforced with [`ruff`](https://docs.astral.sh/ruff/),
and types are checked with `mypy`:
```bash
make format     # apply formatting and autofixes
make lint       # check formatting and lint rules
make typecheck  # run mypy
```

### Documentation

To preview documentation locally, from the root project directory execute:
```bash
make docs
```

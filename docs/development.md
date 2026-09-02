# Development

`gsxform` is a small research library. Contributions of any size are welcome.

## Development environment

`gsxform` uses [`uv`](https://docs.astral.sh/uv/) to manage environments and
dependencies and `make` to organize the process.

```bash
git clone https://github.com/armaank/gsxform.git
cd gsxform
make install
```

`make install` creates the virtual environment, installs `gsxform` with the development and documentation dependencies, and installs the `pre-commit` hooks.

Run `make` on its own to list the available targets.
To preview documentation while writing it, `make docs-serve` gives a live-reloading
server

## Code style

Formatting and linting are handled by [`ruff`](https://docs.astral.sh/ruff/), and types
by `mypy --strict`. Both run in `pre-commit` and in CI, so running `make format` before
committing saves a round trip.

New code in `gsxform/` must be fully typed and carry NumPy-style docstrings. Reshaping is
done with `einops` `rearrange`/`repeat` rather than `.view`/`.permute` — please follow the
surrounding style.

## Branching and pull requests

- `main` is the only long-lived branch. Work happens on short-lived branches that open a
  pull request into `main`.
- Releases are tags on `main`; there is no separate development or stable branch.

Please follow the
[NumPy development workflow](https://numpy.org/doc/stable/dev/development_workflow.html)
naming convention for pull requests.

Every pull request runs lint, types, the full test matrix, and a packaging build. The
single required check is `ci-ok`, which aggregates them. Pull requests also get a
documentation preview, linked from the `docs/preview` check.

## Releasing

Releases are manual and infrequent. Merging to `main` never publishes anything.

1. Bump `version` in `pyproject.toml` and update the changelog.
2. Merge that to `main`.
3. Run the **release** workflow from the Actions tab, choosing the docs alias
   (`latest` or `stable`). It deploys the versioned docs, publishes to PyPI, then tags
   the commit and creates a GitHub Release.

## A note on the macOS x86_64 build

`torch` ships no macOS x86_64 wheels after 2.2.2, and that build is compiled against the
numpy 1.x ABI. Three things encode this: marker-split numpy requirements in
`[project.dependencies]`, `required-environments` in `[tool.uv]`, and a committed
`.python-version`. A dedicated `macos-15-intel` CI job guards the pinned path.

Please do not "simplify" this into a global pin — bounding numpy or torch for every
platform would hobble Linux and Apple Silicon users to work around a constraint that only
affects Intel macs.

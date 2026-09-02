# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

The library itself is unchanged; this release modernises how it is built, tested, and
published.

### Changed

- Build system moved from `setup.py` and `requirements*.txt` to
  [`uv`](https://docs.astral.sh/uv/) with a committed `uv.lock` and PEP 735 dependency
  groups. `pyproject.toml` is now the single source of truth.
- Linting and formatting consolidated onto [`ruff`](https://docs.astral.sh/ruff/),
  replacing `black`, `isort`, `flake8`, and `pydocstyle`. `mypy` now runs in `--strict`
  mode.
- Minimum supported Python raised to 3.10. CI covers 3.10–3.13 on Linux, plus Apple
  Silicon and Intel macOS.
- Documentation is now versioned with [`mike`](https://github.com/jimporter/mike): `dev`
  tracks `main` and each release is published under its own version with a selector.
- Releases are published to PyPI via Trusted Publishing, triggered manually rather than
  from a laptop.

### Added

- `gsxform.__version__`.
- Dependency and GitHub Actions updates via Dependabot.
- Documentation previews on pull requests, surfaced as the `docs/preview` check.
- A single aggregate `ci-ok` status check covering lint, types, tests, and packaging.

### Fixed

- `torch.cat` calls passed `axis=` instead of `dim=` in `wavelets.py` and
  `scattering.py`.
- `TightHann.warp_func` was annotated as returning `torch.Tensor`; it returns a
  `scipy.interpolate.interp1d`.
- Documentation built against a stale `mkdocstrings` configuration and had not been
  redeployed since 2022.

### Platform notes

On macOS x86_64, `torch` is resolved to 2.2.2 and `numpy` to 1.x — the last versions with
wheels for that platform. Every other platform resolves to current releases. See the
development guide for why this is not a global pin.

## [0.1.0]

Initial release, archived at [10.5281/zenodo.7069113](https://doi.org/10.5281/zenodo.7069113).

[Unreleased]: https://github.com/armaank/gsxform/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/armaank/gsxform/releases/tag/v0.2.0
[0.1.0]: https://github.com/armaank/gsxform/releases/tag/v.0.1.0-beta

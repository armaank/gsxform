# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking:** transforms are no longer bound to a graph at construction.
  `Diffusion(W_adj, n_scales, n_layers)(x)` becomes
  `Diffusion(n_scales, n_layers)(x, W_adj)`, and one module now runs over batches of
  any size and node count. For a fixed graph, `cached=True` (as in PyG's `GCNConv`)
  builds the filter bank once and reuses it while the same adjacency is passed.
  `reset_parameters()` clears the cache.
- **Breaking:** the subclass contract is now `get_wavelets(W_adj, mask=None)` and
  `get_lowpass(mask, dtype)`. `TightHann.warp_func` and `get_kernel` take the
  spectrum (and optional mask) as arguments.
- `TightHann` eigendecomposes once per forward instead of once at construction and
  again per forward. `tighthann_wavelets` accepts precomputed `spectra=(E, V)`.

### Added

- torch_geometric integration, installed with `pip install gsxform[pyg]`.
  `gsxform.pyg.GraphScattering` wraps any transform so it takes PyG's
  `(x, edge_index, batch, edge_weight)` and drops into a PyG model.
  `gsxform.pyg.ScatteringFeatures` is a dataset transform that precomputes features
  once per graph.
- `output="node"` returns each scattering path's per-node signal,
  `(batch, n_features, n_coefficients, n_nodes)`, before pooling. It is permutation
  equivariant, for node-level tasks.
- `aggregation="moments"` pools each path with the moments `mean(|path|^q)` for `q`
  in `moments` (default 1–4), following Gao, Wolf & Hirn 2019.
- `path_index()` lists the scale path behind each output coefficient. Coefficients
  are ordered depth-major, then by parent path, then by scale.
- Batches of graphs with different node counts: pad them to a common size (as
  `torch_geometric.utils.to_dense_batch` / `to_dense_adj` do) and pass
  `forward(x, W_adj, mask)`. Each graph's output then matches running it alone. The
  lowpass averages over real nodes only, padded entries of `x` and `W_adj` are
  ignored, and `TightHann` fits its warp on each graph's own eigenvalues.
- Example notebooks started, written as [marimo](https://marimo.io) notebooks
  replacing the empty "Basic Plot" placeholder:
  - `wavelets`: diffusion and tight Hann wavelets drawn on a sensor network, in the
    vertex domain and in the graph's spectrum
  - `qm9_pca`: QM9 scattering embeddings shaded by molecular properties
  - `cuneiform_svm`: Cuneiform sign classification with tight Hann scattering
    features and an RBF SVM
  - `qm9_generation`: a molecule generator with a fixed scattering encoder and a
    trained graph decoder

  The docs build runs each notebook and renders it as a page with its code and
  figures. An examples page explains how to install their dependencies and run them.
  The dependencies, including RDKit, are in a new `examples` dependency group; none
  of them is a dependency of the package itself.

### Fixed

- **Changes outputs:** `TightHann` placed every Hann window one step too high. Its top
  scale passed nothing, so that wavelet and every scattering path through it were
  zero, and the filter bank was not tight: its squared responses fell to half the
  frame constant at the bottom of the spectrum. All scales respond, and `sum_j psi_j^2`
  equals the frame constant`9/8` at every eigenvalue.
- `TightHann` could not be pickled, which broke `torch.save`, multi-worker
  `DataLoader`s and DDP.


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

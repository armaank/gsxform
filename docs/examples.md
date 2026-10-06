# Examples

The examples are [marimo](https://marimo.io) notebooks.

## Setup and Install

The notebooks run from a clone of the repository. Their dependencies are in an
`examples` dependency group. They also need the `pyg` extra.
With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/armaank/gsxform.git
cd gsxform
# run a single example
uv run --extra pyg --group examples marimo edit examples/wavelets.py
```

This opens the notebook in marimo's editor in your browser. To run it as a plain
script instead, use `uv run --extra pyg --group examples python examples/wavelets.py`.

To install the `pyg` extra and the `examples` group into the project's virtual
environment once, instead of passing the flags on every run:

```bash
# install the extra and the examples group into .venv
uv sync --extra pyg --group examples
source .venv/bin/activate
marimo edit examples/wavelets.py
```

The notebooks download their datasets on first run, through torch_geometric, to `data/`
at the repository root. QM9 is the largest.

## Overview

Start with **Understanding graph wavelets and scattering transforms**. It shows what the wavelets and scattering
paths look like on a small graph, and the other notebooks build on its vocabulary.

- **[Understanding graph wavelets and scattering transforms](examples/wavelets.md)**: diffusion and tight Hann wavelets drawn on a sensor network, in the vertex
  domain and in the graph's spectrum.
- **[QM9 embeddings with PCA](examples/qm9_pca.md)**: scattering features of
  QM9 molecules projected with PCA.
- **[Cuneiform classification with an SVM](examples/cuneiform_svm.md)**: tight
  Hann scattering features of Cuneiform signs classified with an RBF SVM.
- **[Generating molecules](examples/qm9_generation.md)**: a VAE with a
  scattering encoder and a trained graph decoder.

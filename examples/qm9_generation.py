import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generating molecules with a scattering VAE

    This notebook trains a generative model of small molecules. Sampling from it produces
    whole new molecules, atoms and bonds together, from nothing but a random latent
    vector.

    Coming soon! 
    """)
    return


@app.cell
def _():
    import time
    import warnings

    # PyG imports tqdm, which warns when notebook progress widgets are unavailable
    warnings.filterwarnings("ignore", message="IProgress not found")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    from torch import nn
    from torch_geometric.data import Data
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import subgraph

    from gsxform import Diffusion
    from gsxform.pyg import GraphScattering

    plt.rcParams["figure.dpi"] = 50  # marimo renders figures at twice this
    ROOT = mo.notebook_dir().parent / "data"
    _ = torch.manual_seed(0)
    rng = np.random.default_rng(0)
    RDLogger.DisableLog("rdApp.*")  # invalid samples would each log an error
    return (
        Chem,
        Data,
        DataLoader,
        Diffusion,
        Draw,
        F,
        GraphScattering,
        PCA,
        QM9,
        ROOT,
        mo,
        nn,
        normalize,
        np,
        plt,
        rng,
        subgraph,
        time,
        torch,
    )


if __name__ == "__main__":
    app.run()

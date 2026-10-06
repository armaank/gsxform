import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualizing QM9 with graph scattering and PCA

    This notebook illustrates how to use scattering transforms to visualize graph data. 

    This notebook embeds 30,000 small molecules from QM9 with a diffusion scattering transform and projects the embeddings onto their first two principal components, shaded by the structural property of each molecule.
    """)
    return


@app.cell
def _():
    import time
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, Lipinski
    from sklearn.decomposition import PCA
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader

    from gsxform import Diffusion
    from gsxform.pyg import GraphScattering

    ROOT = mo.notebook_dir().parent / "data"
    _ = torch.manual_seed(0)
    return (
        Chem,
        DataLoader,
        Descriptors,
        Diffusion,
        GraphScattering,
        Lipinski,
        PCA,
        QM9,
        RDLogger,
        ROOT,
        mo,
        np,
        plt,
        sns,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    QM9 holds about 130k small organic molecules with up to nine heavy atoms (C, N, O, F).
    PyG keeps every hydrogen as a node, so graphs have up to 29 atoms. Molocules in QM9
    are labled with a variety of properties, and can be represented as graphs with atoms as nodes and bonds 
    as edges, or as a string of characters (SMILES).
    """)
    return


@app.cell
def _(QM9, ROOT, torch):
    dataset = QM9(ROOT / "QM9")
    sample = dataset[torch.randperm(len(dataset))[:30_000]]

    print(f"{len(dataset)} molecules, using {len(sample)}")
    print(sample[0])
    return (sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Molecular properties

    The embeddings are shaded by the following properties:

    | Property | Definition |
    |---|---|
    | **HBA** | hydrogen-bond acceptors, RDKit `Lipinski.NumHAcceptors` |
    | **Rings** | ring count, RDKit `Descriptors.RingCount` |
    | **sp3** | fraction of sp3-hybridized carbons, RDKit `Lipinski.FractionCSP3` |

    """)
    return


@app.cell
def _(Chem, Descriptors, Lipinski, RDLogger, np, sample):
    RDLogger.DisableLog("rdApp.error")  # suppress RDKit warnings

    DESCRIPTORS = {
        "HBA": Lipinski.NumHAcceptors,
        "Rings": Descriptors.RingCount,
        "SP3": Lipinski.FractionCSP3,
    }

    molecules = [Chem.MolFromSmiles(data.smiles) for data in sample]
    parsed = [i for i, mol in enumerate(molecules) if mol is not None]
    subset = sample[parsed]
    molecules = [molecules[i] for i in parsed]

    props = {
        name: np.array([describe(mol) for mol in molecules])
        for name, describe in DESCRIPTORS.items()
    }
    print(f"{len(subset)} molecules with properties")
    return props, subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scattering features

    Each atom carries a one-hot element signal over H, C, N, O and F.
    A diffusion scattering transform with 3 scales and 4 layers has
    1 + 3 + 9 + 27 = 40 paths, so each molecule becomes 5 x 40 = 200 coefficients.

    `GraphScattering` runs the transform directly on PyG mini-batches. Molecules of
    different sizes are padded and masked. 
    """)
    return


@app.cell
def _(DataLoader, Diffusion, GraphScattering, subset, time, torch):
    # Define the diffusion scattering transform
    transform = Diffusion(n_scales=3, n_layers=4, nlin=torch.abs)
    # Define the graph scattering operator using the transform
    scattering = GraphScattering(transform)

    start = time.perf_counter()
    with torch.no_grad():
        X = torch.cat(
            [
                scattering(batch.x[:, :5], batch.edge_index, batch.batch)
                for batch in DataLoader(subset, batch_size=1024)
            ]
        ).numpy()

    print(f"features {X.shape} in {time.perf_counter() - start:.1f}s")
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Principal components

    PCA projects the 200-dimensional embedding onto its two directions of largest
    variance. Each point below is one molecule.
    """)
    return


@app.cell
def _(PCA, X, np, plt, props, sns):
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    def embedding_plot(ax, values, title):
        """Scatter plot, shaded by property."""
        order = np.argsort(values, kind="stable")
        ax.scatter(
            Z[order, 0],
            Z[order, 1],
            c=values[order],
            s=4,
            cmap=sns.color_palette("mako", as_cmap=True),
            edgecolors="white",
            linewidths=0.1,
            rasterized=True,
        )
        ax.set_title(title, fontsize=14)
        ax.set_box_aspect(1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 7))
    for ax, (name, values) in zip(axes.flat, props.items()):
        embedding_plot(ax, values, name)
    axes.flat[-1].axis("off")  
    fig.suptitle(
        "Molecular Properties in QM9 visualized via Diffusion scattering"
    )
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()

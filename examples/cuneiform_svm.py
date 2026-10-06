import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classifying cuneiform signs with scattering features and an SVM

    Scattering transforms have no trainable parameters, so a common workflow is to compute
    each graph's scattering coefficients once and hand them to a standard classifier from `scikit-learn`.
    This notebook does that on the Cuneiform dataset, which consists of 267 graphs of handwritten cuneiform signs,
    each one of 30 sign classes (Kriege et al. 2018). Nodes are the wedges a sign is pressed
    from, and edges join touching wedges.

    This examples uses uses tight Hann scattering features and a support vector machine with an RBF
    kernel from `scikit-learn`.
    """)
    return


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")
    warnings.filterwarnings("ignore", message="The number of unique classes")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from torch_geometric.datasets import TUDataset

    from gsxform import TightHann
    from gsxform.pyg import ScatteringFeatures

    plt.rcParams["figure.dpi"] = 50
    ROOT = mo.notebook_dir().parent / "data"
    _ = torch.manual_seed(0)
    return (
        ROOT,
        SVC,
        ScatteringFeatures,
        StandardScaler,
        StratifiedShuffleSplit,
        TUDataset,
        TightHann,
        make_pipeline,
        mo,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    Each wedge carries its 3D position and a one-hot wedge type, 10 features in all.
    Positions are centred within each sign, so the features describe the sign's shape
    rather than where it sits on the tablet. The scattering nonlinearity |x| is not
    translation invariant, so without centring an offset would leak into every
    coefficient.
    """)
    return


@app.cell
def _(ROOT, TUDataset, torch):
    def centre_positions(data):
        """Subtract each sign's mean wedge position from its wedge positions."""
        data = data.clone()
        data.x[:, :3] -= data.x[:, :3].mean(dim=0)
        return data

    dataset = TUDataset(ROOT, "Cuneiform", use_node_attr=True)
    sizes = torch.tensor([data.num_nodes for data in dataset])
    print(dataset, f"{dataset.num_classes} classes")
    print(
        f"wedges per sign: {sizes.min()} to {sizes.max()}, "
        f"mean {sizes.float().mean():.1f}"
    )
    return centre_positions, dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scattering features as a dataset transform

    `ScatteringFeatures` is a PyG dataset transform: it adds a `scattering` attribute to
    every graph, one row of coefficients per graph.
    """)
    return


@app.cell
def _(ScatteringFeatures, TightHann, centre_positions, dataset, np, torch):
    def features(scattering):
        """Stack the scattering features and labels of every sign."""
        transform = ScatteringFeatures(scattering)
        X = torch.cat(
            [transform(centre_positions(data)).scattering for data in dataset]
        ).numpy()
        y = np.array([int(data.y) for data in dataset])
        return X, y

    _X, _y = features(TightHann(4, 3))
    print(
        f"X {_X.shape}, signs per class {np.bincount(_y).min()} to "
        f"{np.bincount(_y).max()}"
    )
    return (features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy against training set size

    Hold out 20% of the signs for testing, train on a growing
    fraction of the rest, and repeat over 20 random stratified splits, for tight Hann
    scattering with 2, 3 and 4 layers. Features are standardized before the SVM.
    """)
    return


@app.cell
def _(
    SVC,
    StandardScaler,
    StratifiedShuffleSplit,
    TightHann,
    dataset,
    features,
    make_pipeline,
    np,
    plt,
):
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    splits = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=0)
    classifier = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    for n_layers in (2, 3, 4):
        X, y = features(TightHann(4, n_layers))
        accuracy = np.zeros((len(fractions), splits.n_splits))
        for split, (pool, test) in enumerate(splits.split(X, y)):
            for ff, fraction in enumerate(fractions):
                n_train = int(round(fraction * len(pool)))
                train = rng.permutation(pool)[:n_train]
                classifier.fit(X[train], y[train])
                accuracy[ff, split] = classifier.score(X[test], y[test])
        ax.errorbar(
            [100 * f for f in fractions],
            accuracy.mean(1),
            yerr=accuracy.std(1),
            marker="o",
            capsize=3,
            label=f"TightHann ({n_layers} layers)",
        )

    ax.axhline(1 / dataset.num_classes, color="gray", ls="--", label="chance")
    ax.set_xlabel("training set used (%)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Cuneiform, RBF SVM on tight Hann scattering features")
    ax.legend()
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()

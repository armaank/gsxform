import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Understanding graph wavelets and scattering transforms

    Coming soon!
    
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.collections import LineCollection

    from gsxform import Diffusion, TightHann, compute_spectra

    rng = np.random.default_rng(1)
    return (
        Diffusion,
        LineCollection,
        TightHann,
        compute_spectra,
        mo,
        np,
        plt,
        rng,
        torch,
    )


if __name__ == "__main__":
    app.run()

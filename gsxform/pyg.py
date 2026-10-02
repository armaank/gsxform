"""Adapters for using scattering transforms in torch_geometric models.

Requires the optional dependency: ``pip install gsxform[pyg]``.
"""

import torch
from einops import rearrange
from torch import nn

try:
    from torch_geometric.data import Data
    from torch_geometric.transforms import BaseTransform
    from torch_geometric.utils import to_dense_adj, to_dense_batch
except ImportError as err:
    raise ImportError(
        "gsxform.pyg requires torch_geometric, install it with "
        "`pip install gsxform[pyg]`"
    ) from err

from .scattering import ScatteringTransform


class GraphScattering(nn.Module):
    """Run a scattering transform on torch_geometric's sparse batch layout.

    Wraps any `ScatteringTransform` so it takes the usual PyG inputs: node
    features `[N_total, F]`, `edge_index`, and the `batch` vector. Graphs of
    different sizes are padded to a common node count and masked internally. 
    """

    def __init__(self, transform: ScatteringTransform) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        transform: ScatteringTransform
            Scattering transform to apply, e.g. `Diffusion(4, 3)`
        """
        super().__init__()
        self.transform = transform

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scattering features for a batch of graphs.

        Parameters
        ----------
        x: torch.Tensor
            Node features, shaped [N_total, F]
        edge_index: torch.Tensor
            Edges, shaped [2, E]
        batch: torch.Tensor, optional
            Graph assignment of each node, shaped [N_total] and sorted, as
            produced by PyG's `DataLoader`. Defaults to None 
        edge_weight: torch.Tensor, optional
            Weight of each edge, shaped [E]. Defaults to None 

        Returns
        -------
        torch.Tensor
            [B, F * n_coefficients] for graph output, or
            [N_total, F * n_coefficients] for node output, feature-major. With
            moment aggregation, each coefficient contributes one column per
            moment.
        """
        batch_size = 1 if batch is None else int(batch.max()) + 1

        x_dense, mask = to_dense_batch(x, batch, batch_size=batch_size)
        W_adj = to_dense_adj(
            edge_index,
            batch,
            edge_weight,
            max_num_nodes=x_dense.shape[1],
            batch_size=batch_size,
        ).to(x.dtype)

        out = self.transform(rearrange(x_dense, "b n f -> b f n"), W_adj, mask)

        if self.transform.output == "node":
            out = rearrange(out, "b f c n -> b n (f c)")
            nodes: torch.Tensor = out[mask]
            return nodes

        graphs: torch.Tensor = rearrange(out, "b f c -> b (f c)")
        return graphs


class ScatteringFeatures(BaseTransform):  # type: ignore[misc]
    """Precompute scattering features as a torch_geometric dataset transform.

    Stores the scattering representation of each graph on the `Data` object,
    so a fixed transform is computed once per graph rather than every epoch,

    Graph output is stored shaped [1, D], so `DataLoader` batches it to [B, D];
    node output is stored shaped [N, D].

    """

    def __init__(self, transform: ScatteringTransform, attr: str = "scattering"):
        """Initialize the dataset transform.

        Parameters
        ----------
        transform: ScatteringTransform
            Scattering transform to apply, e.g. `Diffusion(4, 3)`
        attr: str
            Name of the attribute the features are stored under. Defaults to
            "scattering"
        """
        self.scattering = GraphScattering(transform)
        self.attr = attr

    def forward(self, data: Data) -> Data:
        """Add scattering features to one graph.

        Parameters
        ----------
        data: Data
            Graph with node features `x` and `edge_index`, and optionally
            `edge_weight`

        Returns
        -------
        Data
            The same graph, with the features stored under `attr`
        """
        if data.x is None:
            raise ValueError(
                "ScatteringFeatures needs node features; for featureless graphs "
                "add some first, e.g. torch_geometric.transforms.Constant()"
            )

        with torch.no_grad():
            data[self.attr] = self.scattering(
                data.x,
                data.edge_index,
                edge_weight=getattr(data, "edge_weight", None),
            )

        return data

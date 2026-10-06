"""testing suite for pyg.py"""

import pytest
import torch
from einops import rearrange

pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.nn import GCNConv  # noqa: E402
from torch_geometric.utils import dense_to_sparse  # noqa: E402

from gsxform import scattering  # noqa: E402
from gsxform.pyg import GraphScattering, ScatteringFeatures  # noqa: E402

from .test_utils import pad_graphs  # noqa: E402


def to_data(W_adj: torch.Tensor, x: torch.Tensor) -> Data:
    """Convert one dense graph (n, n) and signal (f, n) to a PyG Data object."""
    edge_index, edge_weight = dense_to_sparse(W_adj)
    return Data(x=x.T, edge_index=edge_index, edge_weight=edge_weight)


def test_adapter_matches_core():  # type: ignore
    """On a PyG batch, the adapter equals the core transform on padded input."""

    torch.manual_seed(0)

    # TightHann on larger graphs only, see test_padded_batch_matches_per_graph
    for sizes, cls in [
        ([9, 17, 24, 12], scattering.Diffusion),
        ([20, 32, 26, 40], scattering.TightHann),
    ]:
        graphs, signals, W_adj, x, mask = pad_graphs(sizes)
        batch = Batch.from_data_list(
            [to_data(g, s) for g, s in zip(graphs, signals, strict=True)]
        )
        inputs = (batch.x, batch.edge_index, batch.batch, batch.edge_weight)

        graph_out = GraphScattering(cls(4, 3))(*inputs)
        expected = rearrange(cls(4, 3)(x, W_adj, mask), "b f c -> b (f c)")
        assert graph_out.shape == (len(sizes), 3 * 21)
        assert torch.allclose(graph_out, expected, atol=1e-5)

        node_out = GraphScattering(cls(4, 3, output="node"))(*inputs)
        expected = rearrange(
            cls(4, 3, output="node")(x, W_adj, mask), "b f c n -> b n (f c)"
        )[mask]
        assert node_out.shape == (sum(sizes), 3 * 21)
        assert torch.allclose(node_out, expected, atol=1e-5)


def test_adapter_single_graph():  # type: ignore
    """With no batch vector the input is one graph."""

    _, _, W_adj, x, _ = pad_graphs([16])
    data = to_data(W_adj[0], x[0])

    out = GraphScattering(scattering.Diffusion(4, 3))(data.x, data.edge_index)

    expected = rearrange(scattering.Diffusion(4, 3)(x, W_adj), "b f c -> b (f c)")
    assert torch.allclose(out, expected, atol=1e-5)


class Model(torch.nn.Module):
    """GCNConv front end, scattering, then a linear readout."""

    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, 8)
        self.scattering = GraphScattering(scattering.Diffusion(3, 2))
        self.readout = torch.nn.Linear(8 * 4, n_classes)

    def forward(self, batch: Batch) -> torch.Tensor:
        h = torch.relu(self.conv(batch.x, batch.edge_index))
        out: torch.Tensor = self.readout(
            self.scattering(h, batch.edge_index, batch.batch)
        )
        return out


def make_dataset(n_graphs: int) -> list[Data]:
    """Small random graphs with random binary labels."""
    graphs, signals, *_ = pad_graphs([10 + ii % 5 for ii in range(n_graphs)])
    dataset = [to_data(g, s) for g, s in zip(graphs, signals, strict=True)]
    for data in dataset:
        data.y = torch.randint(0, 2, (1,))
    return dataset


def test_gradients_reach_upstream_layers():  # type: ignore
    """Gradients flow back through scattering into a preceding PyG layer."""

    torch.manual_seed(0)
    batch = Batch.from_data_list(make_dataset(4))
    model = Model(3, 2)

    torch.nn.functional.cross_entropy(model(batch), batch.y).backward()

    for param in model.conv.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0


def test_model_overfits():  # type: ignore
    """A small model with a scattering layer can fit a handful of graphs."""

    torch.manual_seed(0)
    batch = Batch.from_data_list(make_dataset(8))
    model = Model(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(batch), batch.y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1


def test_scattering_features_batches():  # type: ignore
    """Precomputed features collate to one row per graph."""

    torch.manual_seed(0)
    dataset = make_dataset(8)
    transform = ScatteringFeatures(scattering.Diffusion(4, 3))
    featurized = [transform(data.clone()) for data in dataset]

    batch = next(iter(DataLoader(featurized, batch_size=4)))
    expected = GraphScattering(scattering.Diffusion(4, 3))(
        batch.x, batch.edge_index, batch.batch, batch.edge_weight
    )

    assert batch.scattering.shape == (4, 3 * 21)
    assert torch.allclose(batch.scattering, expected, atol=1e-5)


def test_scattering_features_need_node_features():  # type: ignore
    """Featureless graphs get a pointer to a fix rather than a shape error."""

    data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2)

    with pytest.raises(ValueError, match="node features"):
        ScatteringFeatures(scattering.Diffusion(4, 3))(data)

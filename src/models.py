#!/usr/bin/env python
"""Model definitions for the comparative study.

MLP and GraphSAGE share the same forward signature (x, edge_index) so a
single generic training loop drives both; the MLP simply ignores the graph.
"2 layers" means two learnable layers in both cases.
"""

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Per-cell baseline: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)


class GraphSAGE(nn.Module):
    """SAGEConv -> ReLU -> Dropout -> SAGEConv over the k-NN cell graph."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 dropout: float):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


def build_model(name: str, in_dim: int, n_classes: int, cfg: dict) -> nn.Module:
    tr = cfg["training"]
    cls = {"mlp": MLP,
           "graphsage_pca": GraphSAGE,
           "graphsage_hvg": GraphSAGE}[name]
    return cls(in_dim, tr["hidden_dim"], n_classes, tr["dropout"])

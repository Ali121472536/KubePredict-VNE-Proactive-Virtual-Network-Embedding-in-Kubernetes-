# models/gat_encoder.py

import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class GATEncoder(torch.nn.Module):

    def __init__(self,
                 in_channels=2,
                 hidden_channels=32,
                 out_channels=32,
                 heads=4):

        super(GATEncoder, self).__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False
        )

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)

        return x


# ==========================================================
# Convert NetworkX substrate graph → PyG graph
# ==========================================================

def substrate_to_pyg(substrate_graph):

    edge_index = []

    for u, v in substrate_graph.edges:
        edge_index.append([u, v])
        edge_index.append([v, u])

    edge_index = torch.tensor(edge_index,
                              dtype=torch.long).t().contiguous()

    node_features = []

    for node in substrate_graph.nodes:

        cpu_available = substrate_graph.nodes[node]['cpu_available']

        degree = substrate_graph.degree[node]

        node_features.append([
            cpu_available,
            degree
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    return data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Pooling (optional, depending on your task)
        # For graph-level tasks, consider adding pooling here, e.g.,
        # x = gap(x, data.batch)  # for graph classification/regression

        return x

# Assuming 'num_features' is the dimensionality of your input node features,
# 'hidden_dim' is the size of the hidden layer, and 'output_dim' is the size of the output embeddings.

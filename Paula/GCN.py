import dgl
from dgl.nn import GraphConv
import torch

class GCNModel(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h
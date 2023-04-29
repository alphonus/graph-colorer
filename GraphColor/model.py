import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool

class SimpleNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, 10)
        #self.conv2 = GATv2Conv(4, 2)
        self.classifier = Linear(10, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        #x = self.conv2(x, edge_index)
        #x = x.tanh()
        x = global_mean_pool(x, data.batch)

        out = self.classifier(x)

        return out, x#out, x
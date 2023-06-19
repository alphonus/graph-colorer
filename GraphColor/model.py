import torch
from torch.nn import Dropout
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv, global_mean_pool


class AmazonNet(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(AmazonNet, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim)
        self.dropout = Dropout(p=0.2)
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None):
        #x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)

        return x#out, x

class SimpleNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, 10)
        #self.conv2 = GATv2Conv(4, 2)
        self.dropout = Dropout(p=0.2)
        self.classifier = Linear(10, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = global_mean_pool(x, data.batch)

        out = self.classifier(x)
        out = F.softmax(out, dim=1)
        return out, x#out, x
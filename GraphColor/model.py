import torch
from torch.nn import Dropout
import torch.nn.functional as F
# from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv, global_mean_pool, Linear, BatchNorm


class AmazonNet(torch.nn.Module):
    """
    Based on the Network used in Graph Coloring with Physics-Inspired Graph Neural Networks.
    In the paper they used a 2 Conv layer Network.
    In this approach the Conv was replaced with Transformers.
    """
    def __init__(self, num_features, hidden_dim, num_classes, n_heads=3):
        super(AmazonNet, self).__init__()
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=n_heads)
        self.conv2 = GATv2Conv(hidden_dim*n_heads, hidden_dim, heads=n_heads, concat=False)
        self.dropout = Dropout(p=0.2)
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch=None):
        #x, edge_index = data.x, data.edge_index
        x, edge_index = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x, edge_index = self.conv2(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)

        return x#out, x

class AmazonNet2(torch.nn.Module):
    """
    Based on the Network used in Graph Coloring with Physics-Inspired Graph Neural Networks.
    In the paper they used a 2 Conv layer Network.
    In this approach the Conv was replaced with Transformers.
    """
    def __init__(self, num_features, hidden_dim, num_classes, n_heads=3):
        super(AmazonNet2, self).__init__()
        enc_dim = 32
        self.encoder = Linear(num_features, enc_dim)
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=n_heads, concat=False, dropout=0.2)
        self.norm = BatchNorm(hidden_dim*n_heads)
        self.conv2 = GATv2Conv(hidden_dim*n_heads, hidden_dim, heads=n_heads, concat=False, dropout=0.2)
        self.classifier = Linear(hidden_dim*n_heads, num_classes)

    def forward(self, x, edge_index, batch=None):
        #x, edge_index = data.x, data.edge_index
        #x = self.encoder(x)
        x = self.conv1(x, edge_index) #
        x = self.norm(x)
        #x = x.relu()
        #x = self.dropout(x)
        #x = self.conv2(x, edge_index)
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
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
    def __init__(self, num_features, hidden_dim, num_classes, n_heads=3, chrom_number=21):
        super(AmazonNet2, self).__init__()
        enc_dim = 32
        self.encoder = Linear(num_features, enc_dim)
        self.conv1 = GATv2Conv(num_features, hidden_dim, heads=n_heads, concat=True, dropout=0.2)
        self.norm = BatchNorm(hidden_dim*n_heads)
        self.conv2 = GATv2Conv(hidden_dim*n_heads, hidden_dim, heads=n_heads, concat=False, dropout=0.2)
        self.conv3 = GATv2Conv(hidden_dim, chrom_number, heads=n_heads, concat=False, dropout=0.2)
        self.algo_classifier = Linear(hidden_dim*n_heads, num_classes)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm(x)
        x = self.conv2(x, edge_index)

        color = F.relu(x)
        color = self.conv3(color, edge_index)

        classif = global_mean_pool(x, batch)

        # print("pool", x.shape)
        classif = F.dropout(classif, p=0.3, training=self.training)
        classif = self.algo_classifier(classif)

        return classif, color  # out, x

# helper function for graph-coloring loss
def pots_loss_func(probs, adj_tensor):
    """
    Function to compute cost value based on soft assignments (probabilities)

    :param probs: Probability vector, of each node belonging to each class
    :type probs: torch.tensor
    :param adj_tensor: Adjacency matrix, containing internode weights
    :type adj_tensor: torch.tensor
    :return: Loss, given the current soft assignments (probabilities)
    :rtype: float
    """

    # Multiply probability vectors, then filter via elementwise application of adjacency matrix.
    #  Divide by 2 to adjust for symmetry about the diagonal
    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2

    return loss_

class SimpleNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, chrom_number=21):
        super().__init__()
        self.conv1 = SAGEConv(num_features, 16)
        self.conv2 = SAGEConv(16, 32)
        self.conv3 = SAGEConv(32, chrom_number)
        self.algo_classifier = Linear(32, num_classes)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        color = F.relu(x)
        color = self.conv3(color, edge_index)

        classif = global_mean_pool(x, batch)

        #print("pool", x.shape)
        classif = F.dropout(classif, p=0.3, training=self.training)
        classif = self.algo_classifier(classif)



        return classif, color#out, x
#%%
import importlib
import sys
#import GraphColor
#importlib.reload(sys.modules['GraphColor'])
from GraphColor.dataloader import ColorDataset, load_colform
from GraphColor.model import SimpleNet
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
import os

from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import to_networkx
coloring = dict()
typecast = lambda x: F.one_hot(torch.tensor([x]),num_classes=4)
y_encoding = {'DLMCOL': 0, 'mcs': 1 , 'ILS-TS': 2, 'dsatur':3}
with open('data/raw/export_FIRSTMMDB.csv') as f:
    next(f)
    for line in f:
        file, algo, ncol, time, valid = line.split(',')
        coloring[file] = torch.tensor([y_encoding[algo]])#(algo, int(ncol), float(time), bool(valid))

files = [os.path.join('data/raw', file) for file in os.listdir('data/raw') if file.endswith('.col')]

data_list = [load_colform(file, coloring_file=coloring) for file in files]
loader = DataLoader(data_list, batch_size=2)



#%%
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()
def visualize_embedding(h, color=None, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
G = to_networkx(data_list[0], to_undirected=True)
visualize_graph(G, color=data_list[0].x)#[1]
#%%
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#loader = DataLoader(dataset, batch_size=64, shuffle=False)

model = SimpleNet(1,4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
def train2(loader):
    model.train()
    for data in loader:
        out,_ = model(data)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out,_ = model(data)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

for epoch in range(401):
    train2(loader)
    acc = test(loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {acc:.4f}')
#%% Temp
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
dataset = TUDataset(root='data/TUDataset', name='REDDIT-MULTI-12K')
for i, graph in enumerate(dataset):
    G = to_networkx(graph, to_undirected=True, remove_self_loops=True)
    mapping = dict(zip(G, range(len(G.nodes) + 1)))
    G = nx.relabel_nodes(G, mapping)
    nx.write_edgelist(G, f"data/TUDataset/REDDIT-MULTI-12K/processed/reddit_graph_{i}.edgelist", data=False)
    with open(f"data/TUDataset/REDDIT-MULTI-12K/processed/reddit_graph_{i}.col", 'w') as fileheader:
        fileheader.write(f'p edge {len(G.nodes)} {len(G.edges)}\n')
        for edge in G.edges:
            fileheader.write('e {} {}\n'.format(*tuple(map(lambda x: x + 1, edge))))
#%%
if __name__=='__main__':
    pass
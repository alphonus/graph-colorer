#%%
#%load_ext autoreload

#%autoreload 2
from GraphColor.dataloader import ColorDataset, load_colform
from GraphColor.model import *
from torch_geometric.loader import DataLoader
import torch_geometric.transforms
import torch
import torch.multiprocessing as mp
from functools import partial
import torch.nn.functional as F
import os


#%%
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

#min_size = lambda data, n: data.x.shape[0] > n
def min_size(data, n):
    return data.x.shape[0] > n
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
#G = to_networkx(data_list[0], to_undirected=True)
#visualize_graph(G, color=data_list[0].x)#[1]
#%%
from torch_geometric.datasets import TUDataset
import numpy as np
#dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#loader = DataLoader(dataset, batch_size=64, shuffle=False)


def train(loader, loader_test, epochs):
    for epoch in range(epochs):
        train_sub(loader)
        if epoch % 5 == 0:
            acc = test(loader_test)
            print(f'Epoch: {epoch:03d}, Train Acc: {acc:.4f}')

def train_sub(loader):
    model.train()
    idx = torch.tensor(0, device=device)
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        y = torch.flatten(torch.index_select(torch.reshape(data.y, (-1,3)), 1, idx))
        try:
            loss = criterion(out, y)  # Compute the loss solely based on the training nodes.
        except ValueError:
            raise ValueError("wrong dimensions "
                             f"prediction shape: {out.shape} "
                             f"truth: {y}\n"
                             f"truth: {data.y}\n"
                             f"truth: {data.y.shape}\n"
                             f"input: {data.x.shape}\n"
                             f"truth: {data}\n"
                             f"batch size: {data.batch}\n"
                             f"batch size: {torch.max(data.batch)+1}\n"
                             f"batch size: {data.num_graphs}\n"
                             f"Graphs: {data.name}")
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()

def test(loader):
     model.eval()
     #idx = np.array([i for i in range(loader.batch_size * 3) if i % 3 == 0])
     idx = torch.tensor(0, device=device)
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         y = torch.flatten(torch.index_select(torch.reshape(data.y, (-1, 3)), 1, idx))
         correct += int((pred == y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


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
#if __name__=='__main__':
NUM_PROCESSES = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pre_transforms = torch_geometric.transforms.Compose(
    [torch_geometric.transforms.ToUndirected()])
transforms = torch_geometric.transforms.Compose(
    [torch_geometric.transforms.ToDevice(device)])

# def min_size(n, data):
#    1
#    return data.x.shape[0] > n



filters = partial(min_size, n=50)  #curry the funtion to keep graphs with more than 50 nodes
# torch_geometric.transforms.ComposeFilters([partial(min_size, n=50)])
# length no filter 11929

from numpy.random import default_rng
import math
data_list = ColorDataset(root='data', pre_transform=pre_transforms, transform=transforms, pre_filter=filters)

rng = default_rng()
choice = rng.permutation(len(data_list))
idx = math.floor(len(data_list)*0.8)
train_set = data_list[0:idx]
test_set = data_list[idx:-1]


coloring = dict()
# typecast = lambda x: F.one_hot(torch.tensor([x]),num_classes=4)
y_encoding = {'DLMCOL': 0, 'mcs': 1, 'ILS-TS': 2, 'dsatur': 3, 'hybrid-dsatur': 4, 'head': 5, 'tabucol': 6,
              'lmxrlf': 7}
# with open('data/raw/export_FIRSTMMDB.csv') as f:
#    next(f)
#    for line in f:
#        file, algo, ncol, time, valid = line.split(',')
#        coloring[file] = torch.tensor([y_encoding[algo], int(ncol), bool(valid)])#(algo, int(ncol), float(time), bool(valid))

# files = [os.path.join('data/raw', file) for file in os.listdir('data/raw') if file.endswith('.col')]

# data_list = [load_colform(file, coloring_file=coloring).to(device) for file in files] #+ [load_colform(file, coloring_file=coloring, gen_fake=True) for file in files]
loader = DataLoader(train_set, batch_size=4096, shuffle=True, num_workers=0, pin_memory=False)
loader_test = DataLoader(test_set, batch_size=4096, shuffle=True, num_workers=0, pin_memory=False)
#%%
print("...Creating Model...")
model = AmazonNet(1, 32, loader.dataset.num_classes)

model.to(device)
#model.share_memory()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
print("...Start Training...")
processes = []
#for rank in range(NUM_PROCESSES):
#    p = mp.Process(target=train, args=(loader,100))
#    p.start()
#    processes.append(p)
#for p in processes:
#    p.join()
train(loader, loader_test, 50)


#%% Analys

from torch_geometric.explain import Explainer, GNNExplainer
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='object',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='probs',  # Model returns probabilities.
    ),
)
data = data_list[0]
explanation = explainer(data.x, data.edge_index)
print(explanation.edge_mask)
print(explanation.node_mask)
#explanation.visualize_feature_importance(top_k=10)
#explanation.visualize_graph()
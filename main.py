#%%
#%load_ext autoreload

#%autoreload 2
#%%
from GraphColor.dataloader import ColorDataset, ColorMultiDataset, RandColoring, ColoringOneHot
from GraphColor.model import *
from torch_geometric.nn.models import GAT
from torch_geometric.loader import DataLoader
import torch_geometric.transforms
import torch
import torch.multiprocessing as mp
from functools import partial
import torch.nn.functional as F
import os
#from torch.utils.tensorboard import SummaryWriter
import wandb

#%%
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

hypers = {
    'num_features': 8,
    'embedding_dim': 32

}



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

#%% Process Data
#if __name__=='__main__':
NUM_PROCESSES = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
pre_transforms = torch_geometric.transforms.Compose(
    [torch_geometric.transforms.ToUndirected()]) #, ColoringOneHot(hypers['num_features'], cat=False), RandColoring(hypers['num_features'])
transforms = torch_geometric.transforms.Compose(
   [ torch_geometric.transforms.ToDevice(device)])
"""
pre_transforms = torch_geometric.transforms.Compose(
    [torch_geometric.transforms.ToUndirected(), RandColoring(hypers['num_features'])]) #, ColoringOneHot(hypers['num_features'], cat=False), RandColoring(hypers['num_features'])
transforms = torch_geometric.transforms.Compose(
   [ torch_geometric.transforms.ToDevice(device)])
# def min_size(n, data):
#    1
#    return data.x.shape[0] > n



filters = partial(min_size, n=50)  #curry the funtion to keep graphs with more than 50 nodes
# torch_geometric.transforms.ComposeFilters([partial(min_size, n=50)])
# length no filter 11929

from numpy.random import default_rng
import math
graph_dataset = ColorMultiDataset(root='data/', pre_transform=pre_transforms, transform=transforms, pre_filter=filters)
#graph_dataset = ColorDataset(root='data/WS', pre_transform=pre_transforms, transform=transforms)
for i, data in enumerate(graph_dataset):
    try:
        if not data.validate():
            print(f"Error in data entry No:{i} name:{data.name}")
    except ValueError:
        print(f"IndexError in data entry No:{i} name:{data.name}")
        continue

rng = default_rng()
choice = rng.permutation(len(graph_dataset))
idx = math.floor(len(graph_dataset)*0.8)
train_set = graph_dataset[0:idx]
test_set = graph_dataset[idx:-1]


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
loader_train = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
loader_test = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
#%% Train Model
print("...Creating Model...")

config = {
        "learning_rate": 0.02,
        'feature_rep': 'nn.Embedding',
        "architecture": "GAT jk:last",
        "dataset": "reddit",
        "epochs": 75,
        "log_interval": 2,
        #'NUM_ACCUMULATION_STEPS': 8,
        'n_colors': 32,
        **hypers
    } 
wandb.init(
    # set the wandb project where this run will be logged
    project="Graphs-AAS",

    # track hyperparameters and run metadata
    config=config
   
)


model = AmazonNet(config['num_features'], config['embedding_dim'], loader_train.dataset.num_classes, n_heads=2)
#model = GAT(config['num_features'], config['embedding_dim'], 3 , loader_train.dataset.num_classes, jk=None)
#model = SimpleNet(1, loader_train.dataset.num_classes)
model.to(device)

wandb.watch(model, log_freq=10)
#model.share_memory()
from itertools import chain

embed = torch.nn.Embedding(config['n_colors'], config['num_features'])
embed.to(device)
params = chain(model.parameters(), embed.parameters())

optimizer = torch.optim.AdamW(params, lr=config['learning_rate'])
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()
#criterion = torch.nn.KLDivLoss(reduction='batchmean')
print("...Start Training...")
processes = []
#for rank in range(NUM_PROCESSES):
#    p = mp.Process(target=train, args=(loader,100))
#    p.start()
#    processes.append(p)
#for p in processes:
#    p.join()

hash_tensor = torch.vmap(lambda x: x % config['n_colors'])

def train_single(epoch):
    model.train()
    #idx = torch.tensor(0, device=device)
    loss_all = 0
    for data in loader_train:
        """
        input = torch.squeeze(embed(hash_tensor(data.x.long())))
        out = model(input, data.edge_index)  # , data.batch Perform a single forward pass.
        """
        out = model(data.x, data.edge_index, data.batch) 
        out = F.softmax(out, dim=1)

        loss = criterion(out, data.y) 


        loss_all += loss.item() * data.num_graphs
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {'cat_loss': loss_all / len(train_set),'pots_loss': 0}

def train_pots(epoch):
    model.train()
    #idx = torch.tensor(0, device=device)
    loss_all = 0
    total_pots_loss = 0
    for data in loader_train:

        try:
            
            input = torch.squeeze(embed(hash_tensor(data.x.long())))
            out, color = model(input, data.edge_index, batch=data.batch)  # Perform a single forward pass.
            #print(out.shape)
            #out = F.softmax(out, dim=1)
            color = F.softmax(color, dim=1)
            adj = torch_geometric.utils.to_dense_adj(data.edge_index, data.batch, max_num_nodes=data.num_nodes)
            #print(out)

        except RuntimeError as e:
            print(f"Error using data {data}")
            raise e
        #y = torch.flatten(torch.index_select(torch.reshape(data.y, (-1, 3)), 1, idx))
        #print(y.shape)
        #print(y)

        try:
            loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.

        except ValueError:
            raise ValueError("wrong dimensions \n"
                             f"prediction shape: {out.shape}\n"
                             #f"truth: {y}\n"
                             f"truth: {data.y}\n"
                             f"truth: {data.y.shape}\n"
                             f"input: {data.x.shape}\n"
                             f"truth: {data}\n"
                             # f"batch size: {data.batch}\n"
                             f"batch size: {torch.max(data.batch) + 1}\n"
                             f"batch size: {data.num_graphs}\n"
                             f"Graphs: {data.name}")
        try:
            loss_pots = pots_loss_func(color, adj)
        except RuntimeError as e:
            print("color", color.shape)
            print("adj", adj.shape)
            print("data", data.x.shape)
            raise e

        optimizer.zero_grad()
        (loss + loss_pots).backward()
        #loss_pots.backward()
        loss_all += loss.item() * data.num_graphs
        total_pots_loss += loss_pots * data.num_graphs
        optimizer.step()
    return {'cat_loss': loss_all / len(train_set), 'pots_loss': total_pots_loss/ len(train_set)}

def test(loader):
    model.eval()
    correct = 0
    
    for data in loader:
        #data = data.to(device)
        """
        input = torch.squeeze(embed(hash_tensor(data.x.long())))
        output = model(input, data.edge_index) #, data.batch
        """
        output = model(data.x, data.edge_index, data.batch) 
        output = F.softmax(output, dim=1)
        #color = F.softmax(color, dim=1)
        

        #adj = torch_geometric.utils.to_dense_adj(data.edge_index, data.batch, max_num_nodes=data.num_nodes)
        loss_pots = 0#pots_loss_func(color, adj)
        # selects the index of the max value
        pred = output.max(dim=1)[1]
        #truth = y = torch.flatten(torch.index_select(torch.reshape(data.y, (-1, 3)), 1, idx))
        correct += pred.eq(data.y).sum().item()
        
    return {'acc': correct / len(loader.dataset),'pots_loss':0}

for epoch in range(1, config['epochs']):
    train_loss = train_single(epoch)
    train_acc = test(loader_train)
    test_acc = test(loader_test)
    print('Epoch: {:03d}, Train Cat_Loss: {:.7f}, Train Pots_Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss['cat_loss'], train_loss['pots_loss'],
                                                       train_acc['acc'], test_acc['acc']))
    if epoch % config['log_interval'] == 0:
        wandb.log({
        "train / cat_loss": train_loss['cat_loss'],
        "train / pots_loss": train_loss['pots_loss'],
        "train / acc": train_acc['acc'],
        #"test / pots_loss": test_acc['pots_loss'],
        "test / acc": test_acc['acc']
        })

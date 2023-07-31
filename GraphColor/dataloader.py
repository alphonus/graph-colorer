from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
import torch_geometric.transforms as T
import torch.nn.functional as F

import random
import os #.path as osp
import torch
import copy
from typing import Any
import numpy as np

class RandColoring(BaseTransform):
    def __init__(
        self,
        max_color: int,
        cat: bool = False
    ):
        self.max_color = max_color
        self.cat = cat
        
    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    def gen_density(self, data: Data) -> Data:
        x = data.x
        def rand_vals(seed):
            return torch.randint(int(seed)+1,(1, self.max_color))
        rand = []
        for seed in np.nditer(x.reshape(-1).numpy()):
            rand.append(rand_vals(seed))
        rand = torch.vstack(rand).float()
        #rand = F.softmax(rand, dim=1)
        rand = torch.nn.functional.layer_norm(rand, (self.max_color,))
        return rand
    
    def forward(self, data: Data) -> Data:
        rand = self.gen_density(data)
        x = data.x
        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, rand.to(x.dtype)], dim=-1)
        else:
            data.x = rand

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_color})'


class ColoringOneHot(RandColoring):

    def forward(self, data: Data) -> Data:
        rand = self.gen_density(data)
        rand = torch.argmax(rand, dim=1)
        rand = F.one_hot(rand, self.max_color).long()
        x = data.x
        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, rand.to(x.dtype)], dim=-1)
        else:
            data.x = rand

        return data


def load_colform(file_path, coloring_file=None, train=False, gen_fake=False, dataset: str=''):
    '''
    Loads files in the DIMACS Colro Format
    :param file_path: fully qualified path of the graph file
    :param coloring_file: dictionary of the true colorings for this batch
    :param train: Wether this dataset is for training proposes or inference
    :return: A graph in torch geometric Data format
    '''
    if coloring_file is None and train:
        raise FileNotFoundError('No ground truth coloring provided.')
    edge_index = []
    check = False
    y = None
    if coloring_file:
        fname = file_path.split('/')[-1]
        y = coloring_file.get(fname)
        n_col = y[1]
        valid = y[2]
        y = y[0]
        if y is None:
            raise FileNotFoundError('No coloring for Graph provided.\n'
                                    f"Graph:{dataset+fname} has no coloring\n")
    with open(file_path) as filepointer:

        for line in filepointer:
            if not check and line.startswith('p'):
                n_vertex = int(line.split()[2])
                check = True
            if check and line.startswith('e'):
                edge = line.split()
                if int(edge[1]) == int(edge[2]):
                    continue
                edge_index.append([int(edge[1]) - 1, int(edge[2]) - 1])
    if not check:
        raise FileNotFoundError(f'File: {file_path} not conforming to DIMACS format.')
    if gen_fake:
        edge_set = frozenset(map(frozenset, edge_index))
        candidate = [random.randint(0, n_vertex-1), random.randint(0, n_vertex-1)]
        while candidate[0]==candidate[1] or frozenset(candidate) not in edge_set:
            candidate = (random.randint(0, n_vertex - 1), random.randint(0, n_vertex - 1))
        edge_index.append(candidate)
        #y = torch.cat([y[0:-1].clone().detach(), torch.tensor([False])])
        valid = False
        n_col = y[1]
        y = y[0]


    edge_index = torch.tensor(edge_index, dtype=torch.long)
    #generate random embeding for x
    x = torch.randint(n_vertex-1, (n_vertex,1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.T, y=y, n_col=n_col, valid=valid, name=dataset+fname)
    if not data.validate():
        print(f"Error reading file {dataset+fname}")
        data = None
    return data#T.ToUndirected()(


class ColorDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of ColorDataset. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")

        self.data, self.slices, self.sizes = out

    @property
    def raw_file_names(self):
        return ['export.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'


    #@property
    #def num_node_attributes(self) -> int:
    #    return self.sizes['num_node_attributes']

    @property
    def num_classes(self) -> int:
        return self.sizes['num_classes']

    def process(self):
        data_list = []
        files = os.listdir(os.path.join(self.root, 'raw'))
        col_file = [os.path.join(self.root, 'raw', s) for s in files if 'csv' in s][0]
        coloring = dict()
        y_encoding = {'DLMCOL': 0, 'mcs': 1, 'ILS-TS': 2, 'dsatur': 3, 'hybrid-dsatur': 4, 'head': 5, 'tabucol': 6,
                      'lmxrlf': 7}

        with open(col_file) as f:
            next(f)
            for line in f:
                file, algo, ncol, time, valid = line.split(',')
                coloring[file] = torch.tensor(
                    [y_encoding[algo], int(ncol), bool(valid)])  # (algo, int(ncol), float(time), bool(valid))
        for graph_path in [os.path.join(self.root, 'raw', s) for s in files if '.col' in s]:
            try:
                data_list.append(load_colform(graph_path, coloring_file=coloring))
            except FileNotFoundError:
                #print(f"Skipping file {graph_path}")
                continue

        if self.pre_filter is not None:
            # and not self.pre_filter(data):
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        sizes = {
            'num_node_attributes': 1,
            'num_classes': len(y_encoding)
        }
        torch.save((data, slices, sizes), self.processed_paths[0])

class ColorMultiDataset(ColorDataset):
    """
    Version of the Color PyGeometric Dataset, capable of sourcing from multiple different datasets.
    """
    def process(self):
        data_list = []
        datasets_cols = os.listdir(os.path.join(self.root, 'raw'))


        y_encoding = {'DLMCOL': 0, 'mcs': 1, 'ILS-TS': 2, 'dsatur': 3, 'hybrid-dsatur': 4, 'head': 5, 'tabucol': 6,
                      'lmxrlf': 7}
        for dataset_col in datasets_cols:
            coloring = dict()
            dataset = dataset_col.split('_')[1].split('.')[0]
            print(dataset)
            print(os.path.join(self.root, dataset, 'raw'))
            with open(os.path.join(self.root, 'raw', dataset_col)) as f:
                next(f)
                for line in f:
                    file, algo, ncol, time, valid = line.split(',')
                    coloring[file] = torch.tensor(
                        [y_encoding[algo], int(ncol), bool(valid)])  # (algo, int(ncol), float(time), bool(valid))

            for graph in [s for s in os.listdir(os.path.join(self.root, dataset, 'raw')) if '.col' in s]:
                graph_path = os.path.join(self.root, dataset, 'raw', graph)

                try:
                    data = load_colform(graph_path, coloring_file=coloring, dataset=dataset+'_')
                    data.validate()
                    data_list.append(data)
                except FileNotFoundError as e:
                    print(f"Error in file {graph}, {graph_path}")
                    raise e
                except ValueError:
                    print(f"IndexError in data entry name:{data.name}")
                    continue
                finally:
                    data = None

        if self.pre_filter is not None:
            # and not self.pre_filter(data):
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        #print(data_list)

        data, slices = self.collate(data_list)
        sizes = {
            'num_node_attributes': 1,
            'num_classes': len(y_encoding)
        }
        torch.save((data, slices, sizes), self.processed_paths[0])

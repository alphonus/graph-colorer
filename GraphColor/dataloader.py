import os #.path as osp
import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
import torch_geometric.transforms as T
import random



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
        #y = torch.tensor((y[0:-1], False)) #### Because degree increased by 1???
        y = torch.cat([y[0:-1].clone().detach(), torch.tensor([False])])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    #generate random embeding for x
    x = torch.randint(n_vertex-1, (n_vertex,1), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.T, y=y, name=dataset+fname)
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
                    data_list.append(load_colform(graph_path, coloring_file=coloring, dataset=dataset+'_'))
                except FileNotFoundError as e:
                    print(f"Error in file {graph}, {graph_path}")
                    raise e

        if self.pre_filter is not None:
            # and not self.pre_filter(data):
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print(data_list)

        data, slices = self.collate(data_list)
        sizes = {
            'num_node_attributes': 1,
            'num_classes': len(y_encoding)
        }
        torch.save((data, slices, sizes), self.processed_paths[0])

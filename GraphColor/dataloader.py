import os.path as osp
import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
import torch_geometric.transforms as T
import random



def load_colform(file_path, coloring_file=None, train=False, gen_fake=False):
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
            raise FileNotFoundError('No coloring for Graph provided.')
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
        y = (y[0], False) #### Because degree increased by 1???
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    #generate random embeding for x
    x = torch.randint(n_vertex-1, (n_vertex,1), dtype=torch.float)

    return T.ToUndirected()(Data(x=x, edge_index=edge_index.T, y=y))

class ColorDataset(InMemoryDataset):
    def __init(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform)
        self.data, self. slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data_1.pt']



    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            print(raw_path)
            stop
            data = load_colform(raw_path)
            #data = self.load_colform()
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    #def len(self):
    #    return len(self.processed_file_names)
    #def get(self):
    #    data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
    #    return data
import torch
import pickle
import os
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_sparse import coalesce

def load_dataset(path='../data/raw_data/', dataset='cradle',
                 node_feature_path="../data/cradle/node-embeddings-cradle",
                 num_node=7423):
    print(f'Loading hypergraph dataset from: {dataset}')

    # Load edge labels
    df_labels = pd.read_csv(os.path.join(path, dataset, 'edge-labels-cradle.txt'), sep=',', header=None)
    labels = df_labels.values

    # Load node features
    features = np.zeros((num_node, 128))  # Assuming embedding dimension of 128
    with open(node_feature_path, 'r') as f:
        for line in f:
            values = list(map(float, line.split()))
            features[int(values[0]) - 1] = np.array(values[1:])

    print(f'number of nodes: {num_node}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)

    # Load hyperedge list
    edge_index = []
    with open(os.path.join(path, dataset, 'hyperedges-cradle.txt'), 'r') as f:
        for line in f:
            nodes = list(map(int, line.split(',')))
            edge_index.extend([(node, idx + num_node) for idx, node in enumerate(nodes)])

    edge_index = torch.tensor(edge_index).t().contiguous()
    data = Data(x=features, edge_index=edge_index, y=labels)

    # Coalesce to remove duplicates and sort
    data.edge_index, _ = coalesce(data.edge_index, None, num_node, num_node)

    return data

class CradleDataset(InMemoryDataset):
    def __init__(self, root='../data/pyg_data/cradle_dataset/', transform=None, pre_transform=None):
        super(CradleDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Assume the data is pre-downloaded
        pass

    def process(self):
        # Load and process data here
        data = load_dataset()
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save((data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

#!/usr/bin/env python
# coding: utf-8

import torch
import pickle
import os
import os.path as osp
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_sparse import coalesce


def load_dataset(path='../data', 
                 node_feature_path="../data/node-embeddings-ukb",
                 num_node=3957):

    print('Loading ukb hypergraph dataset')

    # Load edge labels
    df_labels = pd.read_csv(osp.join(path, 'edge-labels-ukb.txt'), sep=',', header=None)
    num_edges = df_labels.shape[0]
    labels = df_labels.values

    # Load node features
    with open(node_feature_path, 'r') as f:
        line = f.readline()
        print(line)
        n_node, embedding_dim = line.split(" ")
        features = np.random.rand(num_node, int(embedding_dim))
        for lines in f.readlines():
            values = list(map(float, lines.split(" ")))
            features[int(values[0])] = np.array(values[1:])

    num_nodes = features.shape[0]

    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    # prepare features and labels
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    
    # build hyperedges 
    p2hyperedge_list = osp.join(path, 'hyperedges-ukb.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1

    # Shift node_idx to start with 0
    # ensures compatibility with PyTorch tensor indices
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    # Constructs a bipartite edge index:
    # Each edge connects a node to a hyperedge 
    # The edge index is represented as a list of two lists: 
    # The first list contains source nodes, The second list contains target nodes
    edge_index = [node_list + he_list,
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    # *** create data object
    data = Data(x=features,
                edge_index=edge_index,
                y=labels)

    # Coalesce edge index
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    data.n_x = n_x
    data.num_hyperedges = he_id - num_nodes

    return data


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    '''
    Save data to a pickle file.
    
    Parameters:
    - data: Data to be saved
    - p2root: Root directory for saving
    - file_name: Optional file name (defaults to a preset name)
    
    Returns:
    - Path to the saved pickle file
    '''
    surfix = 'ukb_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    
    return p2he_StarExpan


class dataset_Hypergraph(InMemoryDataset):
    def __init__(self, name=None, root='../data/pyg_data/hypergraph_dataset/', 
                 p2raw=None, 
                 transform=None, 
                 pre_transform=None, 
                 num_nodes=3957):
        
        self.name = name
        
        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')
        
        if not osp.isdir(root):
            os.makedirs(root)
            
        self.root = root
        self.myraw_dir = osp.join(root, self.name, 'raw')
        self.myprocessed_dir = osp.join(root, self.name, 'processed')
        self.num_nodes = num_nodes
        
        super(dataset_Hypergraph, self).__init__(osp.join(root, self.name), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['ukb']

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        if not osp.isfile(p2f):
            tmp_data = load_dataset(path=self.p2raw,
                                    node_feature_path="../data/node-embeddings-ukb", 
                                    num_node=self.num_nodes)
                    
            _ = save_data_to_pickle(tmp_data, 
                                     p2root=self.myraw_dir,
                                     file_name=self.raw_file_names[0])

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return 'ukb()'
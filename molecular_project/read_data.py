import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import helper as hp

NUM_ATOM_TYPES = 118

def convert_nx_to_torch_geometric(graph: nx.Graph, one_hot_enc_atomic_num: bool=False):
    """
    Convert a single NetworkX graph to a PyTorch Geometric Data object.
    """
    node_info = {node: graph.nodes[node] for node in graph.nodes}

    # get the node features and labels
    node_features = []
    node_labels = []
    node_id_order = []
    for node_id, features_and_labels in node_info.items():
        if one_hot_enc_atomic_num:
            atomic_num = features_and_labels['atomic']
            atomic_num_one_hot = np.zeros(NUM_ATOM_TYPES)
            atomic_num_one_hot[atomic_num] = 1
            features_and_labels['atomic'] = atomic_num_one_hot

        node_features.append([
            features_and_labels['atomic'],
            features_and_labels['valence'],
            features_and_labels['formal_charge'],
            features_and_labels['aromatic'],
            features_and_labels['hybridization'],
            features_and_labels['radical_electrons'],
        ])
        node_labels.append([
            features_and_labels['param']['mass'],
            features_and_labels['param']['charge'],
            features_and_labels['param']['sigma'],
            features_and_labels['param']['epsilon'],
        ])
        node_id_order.append(node_id)

    # permute the node features and labels, so that they are in the correct order from the node_id_order
    node_features = np.array(node_features)[np.argsort(node_id_order)]
    node_labels = np.array(node_labels)[np.argsort(node_id_order)]

    # get the edge features
    edge_info = {edge: graph.edges[edge] for edge in graph.edges}
    edge_index_order = []
    edge_attr = []
    for edge, features in edge_info.items():
        edge_index_order.append(edge)
        edge_attr.append([
            features['stereo'],
            features['aromatic'],
            features['conjugated'],
        ])

    # create the PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index_order, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data

def convert_all_nx_to_torch_geometric(graphs: list[nx.Graph]):
    """
    Convert a list of NetworkX graphs to a list of PyTorch Geometric Data objects.
    """
    data_list = []
    for graph in graphs:
        data = convert_nx_to_torch_geometric(graph)
        data_list.append(data)
    return data_list

def read_nx_data_from_file(filename: str):
    """
    Read a file containing a dictionary of NetworkX graphs.
    """
    graph_data = hp.load_data_from_file(filename)
    return list(graph_data.values())

def read_pyg_data_from_file(filename: str):
    """
    Read a file containing a dictionary of NetworkX graphs and convert them to a list of PyTorch Geometric Data objects.
    """
    graph_dict = hp.load_data_from_file(filename)
    return convert_all_nx_to_torch_geometric(list(graph_dict.values()))

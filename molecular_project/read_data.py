import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import helper as hp
import torch.nn.functional as F

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

        node_feature = []

        if one_hot_enc_atomic_num:
            node_feature += list(features_and_labels['atomic'])
        else:
            node_feature += [features_and_labels['atomic']]

        node_feature.extend([
            features_and_labels['valence'],
            features_and_labels['formal_charge'],
            features_and_labels['aromatic'],
            features_and_labels['hybridization'],
            features_and_labels['radical_electrons'],
        ])
        node_features.append(node_feature)
        node_id_order.append(node_id)
        
        if 'param' in features_and_labels:
            node_labels.append([
                # features_and_labels['param']['mass'],
                features_and_labels['param']['charge'],
                features_and_labels['param']['sigma'],
                features_and_labels['param']['epsilon'],
            ])

    # permute the node features and labels, so that they are in the correct order from the node_id_order
    # needs to also work for list features (e.g. one-hot encoded atomic number)
    node_features = np.array(node_features)[np.argsort(node_id_order)]

    if len(node_labels) > 0:
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
    
    if len(node_labels) > 0:
        y = torch.tensor(node_labels, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def convert_all_nx_to_torch_geometric(graphs: list[nx.Graph], one_hot_enc_atomic_num: bool=False):
    """
    Convert a list of NetworkX graphs to a list of PyTorch Geometric Data objects.
    """
    data_list = []
    for graph in graphs:
        data = convert_nx_to_torch_geometric(graph, one_hot_enc_atomic_num)
        data_list.append(data)
    return data_list

# def convert_all_torch_geometric_to_nx(data_list: list[Data]):
#     """
#     Convert a list of PyTorch Geometric Data objects to a list of NetworkX graphs.
#     """
#     graphs = []
#     for data in data_list:
#         graph = nx.Graph()
#         for i, node in enumerate(data.x):
#             atomic_num = np.argmax(node[:NUM_ATOM_TYPES])
#             graph.add_node(i, **{
#                 'atomic': atomic_num,
#                 'valence': int(node[NUM_ATOM_TYPES].item()),
#                 'formal_charge': int(node[NUM_ATOM_TYPES + 1].item()),
#                 'aromatic': int(node[NUM_ATOM_TYPES + 2].item()),
#                 'hybridization': int(node[NUM_ATOM_TYPES + 3].item()),
#                 'radical_electrons': int(node[NUM_ATOM_TYPES + 4].item()),
#             })
#             if len(data.y) > 0:
#                 graph.nodes[i]['param'] = {
#                     'charge': data.y[i][0].item(),
#                     'sigma': data.y[i][1].item(),
#                     'epsilon': data.y[i][2].item(),
#                 }
#         for i, edge in enumerate(data.edge_index.t()):
#             graph.add_edge(edge[0].item(), edge[1].item(), **{
#                 'stereo': int(data.edge_attr[i][0].item()),
#                 'aromatic': int(data.edge_attr[i][1].item()),
#                 'conjugated': int(data.edge_attr[i][2].item()),
#             })
#         graphs.append(graph)
#     return graphs

def read_nx_data_from_file(filename: str):
    """
    Read a file containing a dictionary of NetworkX graphs.
    """
    graph_data = hp.load_data_from_file(filename)
    return list(graph_data.values())

def read_nx_and_smi_data_from_file(filename: str):
    """
    Read a file containing a dictionary of NetworkX graphs and SMILES strings.
    """
    graph_data = hp.load_data_from_file(filename)
    smis, graphs = zip(*graph_data.items())
    return list(graphs), list(smis)

def read_pyg_and_smi_data_from_file(filename: str, one_hot_enc_atomic_num: bool=False):
    """
    Read a file containing a dictionary of NetworkX graphs and SMILES strings, and convert them to a list of PyTorch Geometric Data objects.
    """
    graph_dict = hp.load_data_from_file(filename)
    smis, graphs = zip(*graph_dict.items())
    data_list = convert_all_nx_to_torch_geometric(graphs, one_hot_enc_atomic_num)
    return list(data_list), list(smis)

def read_pyg_data_from_file(filename: str, one_hot_enc_atomic_num: bool=False):
    """
    Read a file containing a dictionary of NetworkX graphs and convert them to a list of PyTorch Geometric Data objects.
    """
    graph_dict = hp.load_data_from_file(filename)
    return convert_all_nx_to_torch_geometric(list(graph_dict.values()), one_hot_enc_atomic_num)

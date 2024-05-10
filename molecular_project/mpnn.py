from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear, ReLU, Sequential as Seq
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

        # return msg

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
    
class MPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()

        # Add layers for edge convolution and GCNConv
        self.layers.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.layers.append(EdgeConv(hidden_channels, hidden_channels))

        # Add the final message passing layer
        self.layers.append(GCNConv(hidden_channels, out_channels))

        # # Add the final linear layer
        # self.layers.append(Linear(out_channels, out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.layers[0](x, edge_index)
        x = F.tanh(x)

        for layer in self.layers[1: -1]:
            x = layer(x, edge_index)
            x = F.relu(x)

        x = self.layers[-1](x, edge_index)

        return x
    
def train_mpnn(data_loader: DataLoader, model: MPNN, optimizer: torch.optim.Optimizer):
    total_loss = 0
    total_samples = 0

    model.train()
    
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += 1

    return total_loss / total_samples

def train_loop(data_list: list, model: MPNN, optimizer: torch.optim.Optimizer, num_epochs: int = 100, batch_size=64, frac_train=0.9, verbose: bool = False):
    for epoch in range(num_epochs):
        # sample data randomly to train, and validate on the rest
        train_data, val_data = train_test_split(data_list, train_size=frac_train)

        train_data = DataLoader(train_data, batch_size=batch_size)
        val_data = DataLoader(val_data, batch_size=batch_size)

        train_loss = train_mpnn(train_data, model, optimizer)
        if verbose:
            val_loss = test_mpnn(val_data, model)
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

def test_mpnn(data_loader: DataLoader, model: MPNN):
    model.eval()

    total_loss = 0
    total_samples = 0

    for data in data_loader:
        with torch.no_grad():
            out = model(data)
            loss = F.mse_loss(out, data.y)
            total_loss += loss.item()
            total_samples += 1

    return total_loss / total_samples

def mpnn_predict(data_loader: DataLoader, model: MPNN):
    model.eval()

    all_predictions = []
    all_true_labels = []
    for data in data_loader:
        with torch.no_grad():
            out = model(data)
        
        all_predictions.append(out)
        all_true_labels.append(data.y)

    # Concatenate predictions and true labels from all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_labels = torch.cat(all_true_labels, dim=0)
    return all_predictions, all_true_labels

def cross_validate(data_loader: DataLoader, model: MPNN, optimizer: torch.optim.Optimizer, num_folds: int = 5, num_epochs: int = 100, verbose: bool = False):
    # in order to optimize the hyperparameters, we will perform a cross-validation
    
    # TODO: need to implement
    pass
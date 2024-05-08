from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn import Linear, ReLU, Sequential as Seq
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree

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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.layers:
            x = layer(x, edge_index)
            x = nn.functional.relu(x)

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

def train_loop(data_loader: DataLoader, model: MPNN, optimizer: torch.optim.Optimizer, num_epochs: int = 100, verbose: bool = False):
    for epoch in range(num_epochs):
        loss = train_mpnn(data_loader, model, optimizer)
        if verbose:
            print(f'Epoch {epoch}, Loss: {loss}')

def test_mpnn(data_loader: DataLoader, model: MPNN):
    model.eval()

    total_loss = 0
    total_samples = 0

    for data in data_loader:
        out = model(data)
        loss = F.mse_loss(out, data.y)
        total_loss += loss.item()
        total_samples += 1

    return total_loss / total_samples

def cross_validate(data_loader: DataLoader, model: MPNN, optimizer: torch.optim.Optimizer, num_folds: int = 5, num_epochs: int = 100, verbose: bool = False):
    # in order to optimize the hyperparameters, we will perform a cross-validation
    
    # TODO: need to implement
    pass
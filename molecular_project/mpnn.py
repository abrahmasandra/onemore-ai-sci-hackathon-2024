from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.0):
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
    
class GCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5)
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Compute the messages.
        messages = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

        # Step 3: Update the node embeddings.
        out = self.lin(messages)

        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        # x_j has shape [E, in_channels]

        # Step 4: Compute messages.
        return x_j
    
class MPNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int=2):
        super().__init__()
        self.layers = nn.ModuleList()

        # Add the first layer
        self.layers.append(GCNConv(in_channels, hidden_channels))

        # Add the intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(AttentionLayer(hidden_channels, hidden_channels))

        # Add the final layer
        self.layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data: Data):
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
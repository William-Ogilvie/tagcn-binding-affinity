import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class GATv2Net(torch.nn.Module):
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, config: dict):
        """
        Args:
            node_feature_dim (int): Input dimension of node features.
            edge_feature_dim (int): Input dimension of edge features.
            config (dict): config dictionary containing hyperparameters  
        """
        super(GATv2Net, self).__init__()

        # Hyperparameters from Config
        self.n_gnn_layers = config["num_gnn_layers"]
        self.head = config["head"]
        self.hidden_dim = config["hidden_dim"]
         
        # Get activation function 
        if config["activation"] == 'relu':
            self.activation = F.relu
        elif config["activation"] == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        self.GNN_layers = nn.ModuleList()
        self.BN_layers = nn.ModuleList()

        # First layer handles raw input features (e.g. 448)
        curr_dim = node_feature_dim

        # do self.n_gnn_layers of GATv2Conv followed by BatchNorm
        for i in range(self.n_gnn_layers):
            # GATv2 output is (hidden_dim * heads)
            self.GNN_layers.append(
                GATv2Conv(curr_dim, self.hidden_dim, heads=self.head, edge_dim=edge_feature_dim)
            )
            self.BN_layers.append(BatchNorm(self.hidden_dim * self.head))
            curr_dim = self.hidden_dim * self.head

        # MLP
        # As concatenate mean pool and max pool (as in AEV-PLIG) the final_dim * 2

        final_gnn_dim = self.hidden_dim * self.head

        self.fc1 = nn.Linear(final_gnn_dim * 2, 1024)
        self.bn_connect1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_connect2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn_connect3 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 1)


    def forward(self, x, edge_index, edge_attr, batch):
        # x: features
        # edge_index: [2, num_edges]
        # batch: [num_atoms] (tells which atom belongs to which molecule)

        # Message passing (GNN layers)
        for layer, bn in zip(self.GNN_layers, self.BN_layers):
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = bn(x)

        # Global pooling
        # concatenate Global Average and Global Max pooling 
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        # MLP
        x = self.activation(self.fc1(x))
        x = self.bn_connect1(x)
        x = self.activation(self.fc2(x))
        x = self.bn_connect2(x)
        x = self.activation(self.fc3(x))
        x = self.bn_connect3(x)

        return self.out(x)
        

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, TAGConv, BatchNorm
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
        
class TAGCNet(torch.nn.Module):
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, config: dict):
        """
        WARNING: in this version of TAGCNet we apply the hidden_dim of the config directly. This means if you want to compare 
        to say a GATv2 with 5 heads it is worth scaling the hidden dim for TAGCN by 5 (possibly!) to keep it fair.
        Args:
            node_feature_dim (int): Input dimension of node features.
            edge_feature_dim (int): Input dimension of edge features.
            config (dict): config dictionary containing hyperparameters  
        """
        super(TAGCNet, self).__init__()

        # Hyperparameters from Config
        self.n_gnn_layers = config["num_gnn_layers"]
        self.K = config["K"] # K here is the number of hops in TAGNet: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TAGConv.html#torch_geometric.nn.conv.TAGConv 
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

        # do self.n_gnn_layers of TAGConv followed by BatchNorm
        for i in range(self.n_gnn_layers):
            self.GNN_layers.append(
                TAGConv(in_channels=curr_dim, out_channels=self.hidden_dim, K=self.K)
            )
            self.BN_layers.append(BatchNorm(self.hidden_dim))
            curr_dim = self.hidden_dim 

        # MLP
        # As concatenate mean pool and max pool (as in AEV-PLIG) the final_dim * 2

        final_gnn_dim = self.hidden_dim 

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

"""
During training of the above models we get considerable overfitting to the test set, for example in one run of AEV-PLIG
we have a pearson of approx 0.98 on the training set and only 0.74 on the validation set.

The models below are an attempt to address some of the potential problems in the above that may be causing this extreme overfitting
where the NN is likely memorising the ligand or protein pocket.

There are a couple problems with the above models:

1. Batch norm is applied after the activation, this can be a problem because ReLU will kill some of the smaller values before the batch norm has 
a chance to normalise the distribution.
2. The MLP is potentially too big, the first layer has 1024 out features, this could be what is allowing the model to overfit and memorise parts of the
training data.

Fixes:

1. Apply batch norm before activation function
2. Add in dropout after each layer
3. Reduce the size of the MLP
"""

class GATv2Net_v2(torch.nn.Module):
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
        self.K = config["K"]
        self.hidden_dim = config["hidden_dim"]
        self.dropout_rate = config["dropout_rate"]
        self.training = config["training"]
         
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

        # do self.n_gnn_layers of TAGConv followed by BatchNorm
        for i in range(self.n_gnn_layers):
            self.GNN_layers.append(
                TAGConv(in_channels=curr_dim, out_channels=self.hidden_dim, K=self.K)
            )
            self.BN_layers.append(BatchNorm(self.hidden_dim))
            curr_dim = self.hidden_dim 

        # MLP, reducded first layer from 1024 to 512
        # As concatenate mean pool and max pool (as in AEV-PLIG) the final_dim * 2

        final_gnn_dim = self.hidden_dim * self.head

        self.fc1 = nn.Linear(final_gnn_dim * 2, 512)
        self.bn_connect1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_connect2 = nn.BatchNorm1d(256) 
        self.out = nn.Linear(256, 1)


    def forward(self, x, edge_index, edge_attr, batch):
        # x: features
        # edge_index: [2, num_edges]
        # batch: [num_atoms] (tells which atom belongs to which molecule)

        # Message passing (GNN layers)
        for layer, bn in zip(self.GNN_layers, self.BN_layers):
            x = layer(x, edge_index, edge_attr)
            x = bn(x)
            x = self.activation(x)
            F.dropout(input=x, p=self.dropout_rate, training=self.training) 

        # Global pooling
        # concatenate Global Average and Global Max pooling 
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        # MLP, heavier regularization
        x = self.activation(self.bn_connect1(self.fc1(x)))
        x = F.dropout(input=x, p=self.dropout_rate, training=self.training) 
        x = self.activation(self.bn_connect2(self.fc2(x)))
        x = F.dropout(input=x, p=self.dropout_rate, training=self.training) 

        return self.out(x)
    

class TAGCNet_v2(torch.nn.Module):
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
        self.dropout_rate = config["dropout_rate"]
        self.training = config["training"]
         
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

        # MLP, reducded first layer from 1024 to 512
        # As concatenate mean pool and max pool (as in AEV-PLIG) the final_dim * 2

        final_gnn_dim = self.hidden_dim * self.head

        self.fc1 = nn.Linear(final_gnn_dim * 2, 512)
        self.bn_connect1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_connect2 = nn.BatchNorm1d(256) 
        self.out = nn.Linear(256, 1)


    def forward(self, x, edge_index, edge_attr, batch):
        # x: features
        # edge_index: [2, num_edges]
        # batch: [num_atoms] (tells which atom belongs to which molecule)

        # Message passing (GNN layers)
        for layer, bn in zip(self.GNN_layers, self.BN_layers):
            x = layer(x, edge_index, edge_attr)
            x = bn(x)
            x = self.activation(x)
            F.dropout(input=x, p=self.dropout_rate, training=self.training) 

        # Global pooling
        # concatenate Global Average and Global Max pooling 
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim = 1)

        # MLP, heavier regularization
        x = self.activation(self.bn_connect1(self.fc1(x)))
        x = F.dropout(input=x, p=self.dropout_rate, training=self.training) 
        x = self.activation(self.bn_connect2(self.fc2(x)))
        x = F.dropout(input=x, p=self.dropout_rate, training=self.training) 

        return self.out(x)
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import pandas as pd
import os
from functools import lru_cache

# We are going to use Dataset over InMemoryDataset from AEV-PLIG, due to lazy loading. 
# our datasets are quite large and so potentially we don't have the space to feed the entire dataset into RAM. 
# So instead we will just load the batches as we need them.
class PDBDataset(Dataset):
    def __init__(self, data_dir: str):
        """ intialise the class, note you will need absolute paths for this!

        Args: 
            data_dir (str): path to the graphs
        """  
        super().__init__()
        path_to_shard_csv = data_dir / "shard_data.csv"      
        self.df = pd.read_csv(path_to_shard_csv) # Read csv to get unique_ids and shards
        self.data_dir = data_dir # Location of graphs
        self.ids = self.df["unique_id"].tolist()
        # Creating mapping between unique ids and shards
        self.shard_map = dict(zip(self.df["unique_id"], self.df["shard"]))       

    @lru_cache(maxsize=2) # Keeps the current shard in RAM
    def _load_shard(self, shard_file):
        file_path = self.data_dir / shard_file
        return torch.load(file_path, weights_only=False)

    def __len__(self):
        """ number of ids in the dataset

        Returns:
            int: length of self.ids
        """        
        return len(self.ids) 

    def get(self, idx: int):
        """ loads the graph corresponding to the passed index
        Args:
            idx (int): index of data sample attempting to load

        Returns:
            Data: the graph stored as a data object, the number of features, the node features, the edge index and the edge attributes (bond types), then finally the pK value for the protein-ligand complex 
        """     
        
        pdb_id = self.ids[idx]
        shard_file = self.shard_map[pdb_id]

        # This will be instant if the shard is already in the cache
        shard_data = self._load_shard(shard_file)

        # Fetch the specific complex from the shard dictionary
        # [uid, graph, pK] structure
        uid, graph, pK = shard_data[pdb_id]
 
        # Convert to Tensors explicitly
        # graph[1] is x, [2] is edge_index, [3] is edge_attr
        x = torch.tensor(np.array(graph[1]), dtype=torch.float) 

        # LongTensor for indicies in PyTorch Geometric
        edge_index = torch.tensor(np.array(graph[2]), dtype=torch.long).T

        edge_attr = torch.tensor(np.array(graph[3]), dtype = torch.float)
        y = torch.tensor(np.array([pK]), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y) 
    
  

import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
import numpy as np
import pandas as pd
import os
from functools import lru_cache
from tqdm import tqdm
import sys

class PDBDataset(InMemoryDataset):
    def __init__(self, data_dir: str, dataset_name: str, transform=None, pre_transform=None):
        self.dataset_name = dataset_name 
        self.data_dir = data_dir
        # Use the csv to find shards:
        path_to_shard_csv = data_dir / "shard_data.csv"      
        self.df = pd.read_csv(path_to_shard_csv)
        self.ids = self.df["unique_id"].tolist()

        super().__init__(root=str(data_dir), transform=transform, pre_transform=pre_transform)

        # Load the processed data (super().__init__ ensures it exists by calling process() if needed)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        # This is the name of the giant collated file
        return [f"{self.dataset_name}.pt"]

    def process(self):
        print("Processing data into memory...")

        data_list = []

        # Get unique shard numbers to avoid reloading the same shard multiple times
        shard_nums = self.df["shard"].unique()

        for s_num in tqdm(shard_nums, desc="Loading Shards", file=sys.stdout):
            shard_path = self.data_dir / f"shard_{s_num}.pt"
            shard_dict = torch.load(shard_path, weights_only=False)

            for pdb_id, contents in tqdm(shard_dict.items(), desc=f"Loading shard {s_num}", file=sys.stdout):

                uid, graph, pK = contents

                # Convert to Tensors explicitly
                # graph[1] is x, [2] is edge_index, [3] is edge_attr
                x = torch.tensor(np.array(graph[1]), dtype=torch.double)

                # LongTensor for indicies in PyTorch Geometric
                edge_index = torch.tensor(np.array(graph[2]), dtype=torch.long).T

                edge_attr = torch.tensor(np.array(graph[3]), dtype=torch.double)
                y = torch.tensor(np.array([pK]), dtype=torch.double)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y) 
    
                data_list.append(data)
        
        # Collate combines Data objects into one huge Tensor + Slices and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


                 


# Actually the below is too slow on the supercomputer due to slow i/o. So we revert back to InMemoryDataset as above
# On the supercomputer we go from one epoch roughly every 15-20 minutes down to 2-3 epochs a second...

# We are going to use Dataset over InMemoryDataset from AEV-PLIG, due to lazy loading. 
# our datasets are quite large and so potentially we don't have the space to feed the entire dataset into RAM. 
# So instead we will just load the batches as we need them.
# class PDBDataset(Dataset):
#     def __init__(self, data_dir: str):
#         """ intialise the class, note you will need absolute paths for this!

#         Args: 
#             data_dir (str): path to the graphs
#         """  
#         super().__init__()
#         path_to_shard_csv = data_dir / "shard_data.csv"      
#         self.df = pd.read_csv(path_to_shard_csv) # Read csv to get unique_ids and shards
#         self.data_dir = data_dir # Location of graphs
#         self.ids = self.df["unique_id"].tolist()
#         # Creating mapping between unique ids and shards
#         self.shard_map = dict(zip(self.df["unique_id"], self.df["shard"]))       

#     @lru_cache(maxsize=2) # Keeps the current shard in RAM
#     def _load_shard(self, shard_file):
#         shard_file_name = f"shard_{shard_file}.pt"
#         file_path = self.data_dir / shard_file_name
#         return torch.load(file_path, weights_only=False)

#     def len(self):
#         """ number of ids in the dataset

#         Returns:
#             int: length of self.ids
#         """        
#         return len(self.ids) 

#     def get(self, idx: int):
#         """ loads the graph corresponding to the passed index
#         Args:
#             idx (int): index of data sample attempting to load

#         Returns:
#             Data: the graph stored as a data object, the number of features, the node features, the edge index and the edge attributes (bond types), then finally the pK value for the protein-ligand complex 
#         """     
        
#         pdb_id = self.ids[idx]
#         shard_file = self.shard_map[pdb_id]

#         # This will be instant if the shard is already in the cache
#         shard_data = self._load_shard(shard_file)

#         # Fetch the specific complex from the shard dictionary
#         # [uid, graph, pK] structure
#         uid, graph, pK = shard_data[pdb_id]
 
#         # Convert to Tensors explicitly
#         # graph[1] is x, [2] is edge_index, [3] is edge_attr
#         x = torch.tensor(np.array(graph[1]), dtype=torch.float) 

#         # LongTensor for indicies in PyTorch Geometric
#         edge_index = torch.tensor(np.array(graph[2]), dtype=torch.long).T

#         edge_attr = torch.tensor(np.array(graph[3]), dtype = torch.float)
#         y = torch.tensor(np.array([pK]), dtype=torch.float)

#         return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y) 
    
  

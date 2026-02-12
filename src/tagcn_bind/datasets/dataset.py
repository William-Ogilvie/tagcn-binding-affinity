import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import pandas as pd
import os

# We are going to use Dataset over InMemoryDataset from AEV-PLIG, due to lazy loading. 
# our datasets are quite large and so potentially we don't have the space to feed the entire dataset into RAM. 
# So instead we will just load the batches as we need them.
class PDBDataset(Dataset):
    def __init__(self, csv_path: str, data_dir: str):
        """ intialise the class, note you will need absolute paths for this!

        Args:
            csv_path (str): path to csv outlining the dataset
            data_dir (str): path to the graphs
        """        
        self.df = pd.read_csv(csv_path) # Read csv to get unique_ids
        self.data_dir = data_dir # Location of graphs
        self.ids = self.df["unique_id"].tolist()       

    def __len__(self):
        """ number of ids in the dataset

        Returns:
            int: length of self.ids
        """        
        return len(self.ids) 
    
    def __getitem__(self, idx: int):
        """ loads the graph corresponding to the passed index
        Args:
            idx (int): index of data sample attempting to load

        Returns:
            Tuple[List[np.ndarray], List[List[int]], List[List[float]], int]: the graph stored as a tuple, the number of features, the node features, the edge index and the edge attributes (bond types), then finally the pK value for the protein-ligand complex 
        """        

        # Translate the index (42) to a PDB id (e.g. 1a2b)
        pdb_id = self.ids[idx]
        file_path = os.path.join(self.data_dir, f"{pdb_id}_graph.pt")

        # Open the file on disk and pull it into RAM
        uid, graph, pK = torch.load(file_path, weights_only=False)

        # Convert to Tensors explicitly
        # graph[1] is x, [2] is edge_index, [3] is edge_attr
        x = torch.tensor(np.array(graph[1]), dtype=torch.float) 

        # LongTensor for indicies in PyTorch Geometric
        edge_index = torch.tensor(np.array(graph[2]), dtype=torch.long).T

        edge_attr = torch.tensor(np.array(graph[3]), dtype = torch.float)
        y = torch.tensor(np.array([pK]), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
   

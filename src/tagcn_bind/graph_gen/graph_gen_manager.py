"""
graph_gen_manager.py
=====================
This module contains the GraphGenerationManager class which allows us to leverage the GraphGenerator class to process entire datasets (like say PDBbind) in similar style to process_and_predict.py of AEV-PLIG. 
"""

import pandas as pd
import pickle
import torch
import numpy as np
import os
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional 
import traceback
import sys

from .graph_gen import GraphGenerator

def _process_single_row(row: Dict[str, Any], generator: GraphGenerator, atom_keys: pd.DataFrame, atom_map: pd.DataFrame) -> Optional[Any]:
    """ Worker function to process single graph from row using an instance of GraphGenerator.

    Args:
        row (Dict[str, Any]): dictionary containing the unique_id for the graph, and sdf, pdb file paths 
        generator (GraphGenerator): an instance of the GraphGenerator class to make the graphs
        atom_keys (pd.DataFrame): the dataframe used to map between ATOM_TYPE and PDB_ATOM, for the pdb
        atom_map (pd.DataFrame): the atom map that maps ATOM_TYPEs of the pdb into dummy encoded "atomic numbers"

    Returns:
        Optional[Any]: either returns a tuple of the unique graph id, the graph itself and the complex binding affinity pK, or returns None if there has been an error 
    """     

    unique_id = row.get("unique_id")
    pdb_path = row.get("pdb_file")
    sdf_path = row.get("sdf_file")
    pK = row.get("pK")

    try:
        # 1. Load Ligand
        sdf_mol = generator.load_sdf(sdf_path)
        if sdf_mol is None:
            return None

        # 2. Load Protein
        pdb_df = generator.load_pdb(pdb_path, atom_keys)

        # 3. Convert Ligand to DF
        sdf_df = generator.mol_to_df(sdf_mol)

        # 4. Get AEVs
        ligand_df, final_aevs = generator.get_mol_aevs(pdb_df, sdf_df, atom_map)

        # 5. Build Graph
        graph = generator.mol_to_graph(sdf_mol, final_aevs, ligand_df)

        return unique_id, graph, pK

    except Exception as e: 
        print(f"Error processing {unique_id}:")
        traceback.print_exc()
        return None
       


class GraphGenerationManager:
    def __init__(self, config: Dict[str, Any], atom_keys_path: str):
        """
        Args:
            config (dict): Configuration dictionary containing graph_generator_coefs.
            atom_keys_path (str): Path to PDB_Atom_Keys.csv.
        """
        self.config = config
        
        # Load Atom Keys
        self.atom_keys = pd.read_csv(atom_keys_path, sep=",")
        
        # Extract Coefs
        self.coefs = config["graph_generator_coef"]
        
        # Initialize a temporary generator just to create the atom_map
        # (We will create the actual generator inside workers or loop)
        temp_gen = GraphGenerator(
            RcR=self.coefs["RcR"],
            EtaR=torch.tensor(self.coefs["EtaR"]),
            RsR=torch.tensor(self.coefs["RsR"]),
            legacy_mode=self.coefs["legacy_mode"],
            ligand_bins=self.coefs["ligand_bins"],
            use_og_torchani=self.coefs["use_og_torchani"],
            allowed_elts=self.coefs["allowed_elts"],
            device="cpu"
        )
        self.atom_map = temp_gen.create_atom_map(self.atom_keys)

    def generate_graphs(self, data_csv_path: str, output_path: str, device: str = "cpu"):
        """
        Generates graphs from a CSV file sequentially.

        Args:
            data_csv_path (str): Path to the CSV containing 'unique_id', 'sdf_file', 'pdb_file'.
            output_path (str): Path to save the resulting pt files. must be passed as an abosulte path for your file system
            device (str): 'cpu' or 'cuda'.
        """
        df = pd.read_csv(data_csv_path)
        rows = df.to_dict("records")
        

        # Prepare the generator configuration
        # We instantiate the generator here. 

        generator = GraphGenerator(
            RcR=self.coefs["RcR"],
            EtaR=torch.tensor(self.coefs["EtaR"]),
            RsR=torch.tensor(self.coefs["RsR"]),
            legacy_mode=self.coefs["legacy_mode"],
            ligand_bins=self.coefs["ligand_bins"],
            use_og_torchani=self.coefs["use_og_torchani"],
            allowed_elts=self.coefs["allowed_elts"],
            device=device
        )

        start_time = time.time()
        graph_counter = 0
        shard_counter = 0
        shard_index = 0

        # On the supercomputer there is a bottle neck with I/O time, to solve this
        # we are going to save the graphs in blocks of 1,000 shards and then just load the shards
        shard = dict()
        shard_data = []

        print(f"Starting sequential graph generation on {device}...")
        for row in tqdm(rows, desc="Generating Graphs", file=sys.stdout):
            result = _process_single_row(row, generator, self.atom_keys, self.atom_map)
            if result is not None:
                uid, graph, pK = result

                print(f"Saving graph to {output_path} as {uid}_graph.pt, and to shard {shard_index}.")
                torch.save((uid, graph, pK), f"{output_path}/{uid}_graph.pt")
                shard[uid] = (uid, graph, pK) 
                shard_counter += 1
                graph_counter += 1
                shard_data.append({"unique_id": uid, "shard": shard_index})

                # If we have 1,000 elements in the shard we will save it:
                if shard_counter == 1000:
                    print(f"Saving shard {shard_index}")
                    torch.save(shard, f"{output_path}/shard_{shard_index}.pt")
                    shard_index += 1

                    # Reset counter
                    shard_counter = 0
                    shard = dict()
                    

        # Save the final shard 
        print(f"Saving shard {shard_index}")
        torch.save(shard, f"{output_path}/shard_{shard_index}.pt")

        # Save the shard dataframe for loop up
        shard_df = pd.DataFrame(shard_data) 
        shard_df.to_csv(f"{output_path}/shard_data.csv", index=False)  
                    
                

        print(f"Graph generation complete. Processed {graph_counter}/{len(rows)} successfully.")
        print(f"Time taken: {time.time() - start_time:.2f}s")


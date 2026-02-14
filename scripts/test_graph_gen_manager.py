"""
test_graph_gen_manager.py
==========================

This script will test GraphGenerationManager, assuming you have already run: python create_dataset_csv.py --data_dir data/CASF-2016 --sdf_format X_ligand.sdf --pdb_format X_protein.pdb --output_csv data/CASF_2016_processed.csv, to get the processed csv
for CASF-2016. You may also need to make a graphs directory inside the data folder
""" 
from pathlib import Path
import yaml
import torch
import pickle
import numpy as np
import pandas as pd
from tagcn_bind import GraphGenerationManager
import math

def main():
    # Get the absolute path of the current script
    script_path = Path(__file__).resolve()

    # Go up two levels to get the project root:
    project_root = script_path.parent.parent

    # Create data/graphs/ directory if doesn't already exist
    data_graphs_dir = project_root / "data" / "graphs"

    data_graphs_dir.mkdir(parents= True, exist_ok= True)

    # Graph Gen config:
    graph_gen_config_path = project_root / "config" / "graph_gen_config.yml"

    # Load config
    with open(graph_gen_config_path, "r") as f:
            config = yaml.safe_load(f)

    # Get atom_keys for pdb files:
    pdb_atom_keys_path = project_root / "config" / "PDB_Atom_Keys.csv"

    # Path to processed csv
    data_csv_path = project_root / "data" / "CASF_2016_processed.csv"

    # Initialise GraphGenerationManager
    graph_generator_manager = GraphGenerationManager(config=config, atom_keys_path=pdb_atom_keys_path)

    # Run sequentially (num_workers=0) to avoid multiprocessing issues and allow GPU usage
    # This is safer and easier to debug
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running sequentially on {device}...")

    # Set output path
    output_path = data_graphs_dir / "CASF-2016-test"
    output_path.mkdir(parents=True, exist_ok=True)
    
    graph_generator_manager.generate_graphs(
        data_csv_path=data_csv_path, 
        output_path=output_path, 
        device=device
    )

    # --- Verification Section ---
    print("\n--- Verifying Generated Graphs ---")
    
    if not output_path.exists() or not output_path.is_dir():
        print(f"Error: Output directory {output_path} not found!")
        return

    # List all .pt files
    pt_files = list(output_path.glob("*.pt"))
    num_shards = len(pt_files)
    print(f"Found {num_shards}  files in {output_path}.")
    
    # Load CSV to compare
    if data_csv_path.exists():
        df = pd.read_csv(data_csv_path)
        expected_count = math.ceil(len(df) / 1000)
        
        if num_shards == expected_count:
            print(f"SUCCESS: Shard count matches expected ({expected_count}).")
        else:
            print(f"WARNING: Shard count ({num_shards}) does not match expected ({expected_count}).")
    else:
        print(f"WARNING: CSV file {data_csv_path} not found for comparison.")
        df = None

    if num_shards > 0:
        # Inspect first file
        first_shard = pt_files[0]
        print(f"\nInspecting shard: {first_shard.name}")
        
        try:
            # First load the shard
            loaded_shard = torch.load(first_shard, weights_only=False)
            # Load the tuple (unique_id, graph, pK) of the first unique id 
            graph_id = list(loaded_shard.keys())[0]
            uid, graph, pK = loaded_shard[graph_id] 
            print(f"Loaded Unique ID: {uid}")
            print(f"Loaded pK: {pK}")

            # Verify against CSV
            if df is not None:
                # Check if uid exists in df
                row = df[df["unique_id"].astype(str) == str(uid)]
                
                if not row.empty:
                    print("SUCCESS: Unique ID found in CSV.")
                    csv_pk = row.iloc[0].get("pK")
                    
                    if csv_pk is not None and not pd.isna(csv_pk):
                        if abs(float(csv_pk) - float(pK)) < 1e-5:
                            print(f"SUCCESS: pK matches CSV ({csv_pk}).")
                        else:
                            print(f"WARNING: pK mismatch! CSV: {csv_pk}, Graph: {pK}")
                    else:
                        print("Notice: pK not present in CSV row.")
                else:
                    print(f"WARNING: Unique ID {uid} not found in CSV!")

            # Unpack graph structure: (num_atoms, node_features, edge_index, edge_attr, len_chem_feats, len_aev_feats)
            num_atoms, node_features, edge_index, edge_attr, len_chem_feats, len_aev_feats = graph
 
        except Exception as e:
            print(f"ERROR: Failed to inspect graph: {e}")           
 
        print(f"Number of atoms: {num_atoms}")
        print(f"Node features type: {type(node_features)}")
        print(f"Node features length: {len(node_features)}")
        
        if len(node_features) > 0:
            feat_shape = node_features[0].shape
            print(f"Node feature shape (per atom): {feat_shape}")            

            # Check for NaNs 
            all_feats = np.stack(node_features) 
            if np.isnan(all_feats).any():
                print("ERROR: Node features contain NaNs!")
            else:
                print("Node features are clean (no NaNs).")

            # Check for zero vectors (potential AEV failure)
            if not np.any(all_feats):
                print("WARNING: All node features are zero!")
            else:
                print(f"Sample feature vector (first atom, first 200 dims): {node_features[0][:200]}")

        print(f"Number of edges: {len(edge_index)}")
        if len(edge_index) > 0:
            print(f"First edge: {edge_index[0]}")
            print(f"First edge attr: {edge_attr[0]}")
            
        # Verify consistency
        if len(node_features) != num_atoms:
             print(f"ERROR: Mismatch between num_atoms ({num_atoms}) and len(node_features) ({len(node_features)})")
        else:
             print("Consistency Check: num_atoms matches feature length.")

        if len(edge_index) != len(edge_attr):
             print(f"ERROR: Mismatch between edge_index ({len(edge_index)}) and edge_attr ({len(edge_attr)}) length")
        else:
             print("Consistency Check: edge_index matches edge_attr length.")


    print("\nVerification complete.")

if __name__ == "__main__":
    main()
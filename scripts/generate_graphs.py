"""
generate_graphs.py
==================

Script to generate graphs for multiple datasets defined in a configuration file.

Example Config Structure (YAML):
--------------------------------
datasets:
  - name: "CASF-2016"
    data_csv_path: "data/CASF_2016_processed.csv"
    output_dir: "data/graphs/CASF-2016"
    atom_keys_path: "config/PDB_Atom_Keys.csv"
    graph_gen_config_path: "config/graph_gen_config.yml"

Usage:
    python scripts/generate_graphs.py --config_path config/graph_generation.yml --device auto
"""
import argparse
import yaml
import torch
from pathlib import Path
from tagcn_bind import GraphGenerationManager

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Generate graphs for datasets defined in a config file.")
    parser.add_argument("--config_path", type=str, required=True, help="Relative path to the graph generation config file.")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu).")
    return parser.parse_args()

def get_device(device_str):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str

def main():
    args = parse_args()
    device = get_device(args.device)
    
    # Resolve config path
    config_path = project_root / args.config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from {config_path}")
    print(f"Using device: {device}")

    # Iterate over datasets defined in the config
    # We expect a list under the key 'datasets'
    datasets = config.get("datasets", [])
    
    if not datasets:
        print("No datasets found in config (key 'datasets' is empty or missing).")
        return

    for i, dataset in enumerate(datasets):
        name = dataset.get("name", f"Dataset_{i}")
        print(f"\n--- Processing Dataset: {name} ---")

        # Extract paths (relative to project root)
        try:
            data_csv_rel = dataset["data_csv_path"]
            output_dir_rel = dataset["output_dir"]
            atom_keys_rel = dataset["atom_keys_path"]
            graph_gen_config_rel = dataset["graph_gen_config_path"]
        except KeyError as e:
            raise KeyError(f"missing key {e}") 

        # Resolve absolute paths
        data_csv_path = project_root / data_csv_rel
        output_dir = project_root / output_dir_rel
        atom_keys_path = project_root / atom_keys_rel
        graph_gen_config_path = project_root / graph_gen_config_rel

        # Check existence of input files
        if not data_csv_path.exists():
            print(f"Error: Data CSV not found at {data_csv_path}")
            continue
        if not atom_keys_path.exists():
            print(f"Error: Atom keys file not found at {atom_keys_path}")
            continue
        if not graph_gen_config_path.exists():
            print(f"Error: Graph gen config not found at {graph_gen_config_path}")
            continue

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load graph generation config (coefficients)
        with open(graph_gen_config_path, "r") as f:
            graph_gen_config = yaml.safe_load(f)

        # Initialize Manager
        try:
            manager = GraphGenerationManager(config=graph_gen_config, atom_keys_path=str(atom_keys_path))
            
            # Generate Graphs
            manager.generate_graphs(
                data_csv_path=str(data_csv_path),
                output_path=str(output_dir),
                device=device
            )
        except Exception as e:
            print(f"Failed to process dataset {name}: {e}")

if __name__ == "__main__":
    main()
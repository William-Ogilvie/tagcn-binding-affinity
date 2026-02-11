"""
scaling_script.py
==================

Script to generate scaling for the passed datasets stores them in the passed file .

Example Config Structure (YAML):
--------------------------------
datasets:
  - name: "PDBbind_minus_CASF_2016"
    dataset_paths: ["data/graphs/PDBbind_train", "data/graphs/PDBbind_val"]
    output_dir: "data/scaling"
    output_file_name: "PDBbind_minus_CASF_2016_scaling"
    
Usage:
    python scripts/scaling_script.py --config_path config/scaling_generation_PDBbind.yml
"""
import argparse
import yaml
from pathlib import Path
from tagcn_bind import ScalingManager

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Generate scaling stats for datasets defined in a config file.")
    parser.add_argument("--config_path", type=str, required=True, help="Relative path to the scaling generation config file.")
    return parser.parse_args()

def main():
    args = parse_args()
   
    # Resolve config path
    config_path = project_root / args.config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from {config_path}")

    # Iterate over datasets defined in the config
    # We expect a list under the key 'datasets'
    datasets = config.get("datasets", [])
    
    if not datasets:
        print("No datasets found in config (key 'datasets' is empty or missing).")
        return
  

    for i, dataset in enumerate(datasets):
        name = dataset.get("name", f"Dataset_{i}")
        print(f"\n--- Processing Dataset: {name} ---")

        # Extract config
        try:
            dataset_paths = dataset["dataset_paths"] 
            output_dir = dataset["output_dir"]
            output_file_name = dataset["output_file_name"]
        except KeyError as e:
            raise KeyError(f"missing key {e}")
         

        # Resolve absolute paths
        new_dataset_paths = []
        for path in dataset_paths:
            abs_path = project_root / path
            new_dataset_paths.append(abs_path)

        output_dir = project_root / output_dir

        output_dir.mkdir(parents = True, exist_ok = True)

        # Initalise scaling manager
        scaling_manager = ScalingManager()

        # Computer scaling
        scaling_manager.calculate_stats(dataset_paths=new_dataset_paths, output_path=output_dir, output_file_name=output_file_name)
    

if __name__ == "__main__":
    main()
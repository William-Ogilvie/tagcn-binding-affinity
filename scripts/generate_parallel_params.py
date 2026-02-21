"""
generate_parallel_params.py
===========================

Generates a text file containing (experiment_name, seed) pairs for Slurm array jobs.
Reads from the experiments config.

Usage:
    python scripts/generate_parallel_params.py --config_path config/experiments_config.yml --output_path config/parallel_params.txt
"""
import argparse
import yaml
from pathlib import Path
import sys

# Get the absolute path of the current script
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Generate parallel params file for Slurm array jobs.")
    parser.add_argument("--config_path", type=str, required=True, help="Relative path to the experiments config file.")
    parser.add_argument("--output_path", type=str, default="config/parallel_params.txt", help="Relative path to the output text file.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_path = project_root / args.config_path
    output_path = project_root / args.output_path
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    tasks = []
    
    if "experiments" in config:
        for experiment in config["experiments"]:
            exp_name = experiment.get("name")
            seeds = experiment.get("args", {}).get("seeds", [])
            
            for seed in seeds:
                tasks.append(f"{exp_name} {seed}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for task in tasks:
            f.write(f"{task}\n")
            
    print(f"Generated {len(tasks)} tasks in {output_path}")

if __name__ == "__main__":
    main()

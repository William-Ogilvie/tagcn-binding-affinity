#!/bin/bash

#SBATCH --job-name=tagcn_process_Zero_Ligand_Bias_Graphs
#SBATCH --time=03:00:00
#SBATCH --partition=short
#SBATCH --clusters=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge

# Load conda
source /data/stat-cadd/shug7579/miniconda3/etc/profile.d/conda.sh

# Activate env
conda activate tagcn-bind-gpu

# Move to project
cd /data/stat-cadd/shug7579/tagcn-binding-affinity/scripts

# This script will generate the graphs for 0 Ligand Bias

# Generate process csvs
python create_zero_ligand_bias_datasets_csv.py --zero_train_path data/zero_ligand_bias_train.csv --zero_test_path data/zero_ligand_bias_test.csv --pdbbind_dir data/PDBbind --sdf_format X_ligand.sdf --pdb_format X_protein.pdb --output_dir data --split_ratio 0.1 --seed 37 --index_file data/index/INDEX_general_PL_data.2020

# Generate new graphs:
# Generate graphs using the config
python generate_graphs.py --config_path config/graph_generation_zero_ligand_bias.yml --device auto

# Generate scaling stats
python scaling_script.py --config_path config/scaling_generation_zero_ligand_bias.yml
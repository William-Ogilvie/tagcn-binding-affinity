#!/bin/bash
# This bash script will process the PDBbind data, splitting it into a train, validation and test set
# where the test set is CASF-2016, it will do this for both the new and legacy version of the graphs

set -euo pipefail # strict mode
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root is $PROJECT_ROOT"

SCRIPTS="scripts"

cd $PROJECT_ROOT
cd $SCRIPTS

# Create the process CSVs
python create_dataset_csv.py --data_dir data/CASF-2016 --sdf_format X_ligand.sdf --pdb_format X_protein.pdb --output_csv data/CASF_2016_processed.csv --index_file data/index/INDEX_general_PL_data.2020
python create_dataset_csv.py --data_dir data/PDBbind --sdf_format X_ligand.sdf --pdb_format X_protein.pdb --output_csv data/PDBbind_processed.csv --index_file data/index/INDEX_general_PL_data.2020

# Remove CASF-2016 from PDBbind
python remove_dataset_csv.py --input_csv data/PDBbind_processed.csv --remove_csv data/CASF_2016_processed.csv --output_csv data/PDBbind_minus_CASF_2016_processed.csv

# Generate a 10% validation set
python create_val_dataset_csv.py --input_csv data/PDBbind_minus_CASF_2016_processed.csv --val_csv data/PDBbind_processed_val.csv --train_csv data/PDBbind_processed_train.csv --split_ratio 0.1 --seed 37

# Generate new graphs:
# Generate graphs using the config
python generate_graphs.py --config_path config/graph_generation_PDBbind.yml --device auto

# Generate scaling stats
python scaling_script.py --config_path config/scaling_generation_PDBbind.yml

# Generate legacy graphs:
# Generate graphs using the config
python generate_graphs.py --config_path config/graph_generation_PDBbind_legacy.yml --device auto

# Generate scaling stats
python scaling_script.py --config_path config/scaling_generation_PDBbind_legacy.yml


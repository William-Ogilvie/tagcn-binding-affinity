#!/bin/bash

#SBATCH --job-name=tagcn_bind_train_parallel
#SBATCH --time=24:00:00
#SBATCH --array=0-99%10
#SBATCH --partition=medium
#SBATCH --clusters=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge

# Implement the below for your setup
# Load conda
# Activate env
# Move to project

# Generate the parameters for the config
python generate_parallel_params.py --config_path config/experiments_config.yml --output_path config/parallel_params.txt

# Get the parameters for this specific task ID from the file
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" config/parallel_params.txt)

# Split the line into variables
EXP_NAME=$(echo $PARAMS | awk '{print $1}')
SEED=$(echo $PARAMS | awk '{print $2}')

# Run your refactored worker
python training_script_parallel.py --config_path config/experiments_config.yml --experiment_name $EXP_NAME --seed $SEED --device auto   


#!/bin/bash

#SBATCH --job-name=tagcn_bind_process_parallel
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

# Implement the below for your setup
# Load conda
# Activate env
# Move to project
# Get a $PROJECT_ROOT variable

python process_script_parallel.py --config_path config/process_config.yml --device auto

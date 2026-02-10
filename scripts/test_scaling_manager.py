"""
test_trainer.py
=================

This script will test the ScalingManager class, we assume you already have made the data/graphs/CASF-2016-test set from test_graph_gen_manager.py
"""
from pathlib import Path
from tagcn_bind import ScalingManager

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

# Get the directory to the example dataset we will use
dataset_dir = project_root / "data" / "graphs" / "CASF-2016-test"

# Get the directory to store the scaling stats
scaling_dir = project_root / "data" / "scaling"

scaling_dir.mkdir(parents=True, exist_ok=True)

# File name for the scaling stats
scale_file_name = "CASF_2016_test_scaling_stats"


# We now initalise the ScalingManager class
trainer = ScalingManager()

# Calculate stats
trainer.calculate_stats(data_set_path=dataset_dir, output_path=scaling_dir, output_file_name=scale_file_name)



"""
compute_bootstrap.py
=======================

This script will allow for the computing of bootstrap confidence intervals for model performance metrics (PCC, Kendall's Tau) 
as well for the computation of p-values to test for a significant difference between two models performance metric.

Like other scripts we are going to pass a config with all the required paramaters to allow us to process in larger batches.
Structure the configs as follows:
- model_name_1: TAGCN-Drop-0-2-K-2-L-5-Dim-256-WD-0-30-1-Inter-Zero-Ligand-Bias
  time_stamp_1: '20260321_161700'
  model_name_2: ""
  time_stamp_2: ""
  confidence_interval: false
  comparison: false 
  metrics: ["RMSE", "Pearson", "Kendall"]
  seed: 37

"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, kendalltau
import sys

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="This script allows for computation of bootstrap confidence intervals for performance metrics or to test for statistical significance between two models performance metrics.."
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Relative path to the config."
    )     
    return parser.parse_args()

def bootstrap_comparison(m1_preds, m2_preds, targets, seed, n_bootstraps=10000):
    np.random.seed(seed=seed)
    
    deltas_pearson = []
    deltas_kendall = []
    deltas_rmse = []
    n = len(targets)
    
    for _ in range(n_bootstraps):
        # Sample indices with replacement
        idx = np.random.choice(n, n, replace=True)
        
        # Calculate metric for both on the same sample
        rho_m1 = pearsonr(m1_preds[idx], targets[idx])[0]
        rho_m2 = pearsonr(m2_preds[idx], targets[idx])[0]
        tau_m1 = kendalltau(m1_preds[idx], targets[idx])[0]
        tau_m2 = kendalltau(m1_preds[idx], targets[idx])[0]
        rmse_m1 = np.sqrt(np.mean(m1_preds[idx] - targets[idx]))
        rmse_m2 = np.sqrt(np.mean(m2_preds[idx] - targets[idx])) 
        
        deltas_pearson.append(rho_m1 - rho_m2)
        deltas_kendall.append(tau_m1 - tau_m2)
        deltas_rmse.append(rmse_m1 - rmse_m2)

    
    deltas_pearson = np.array(deltas_pearson)
    deltas_kendall = np.array(deltas_kendall)
    deltas_rmse = np.array(deltas_rmse)
    p_value_pearson = np.mean(deltas_pearson <= 0) # Proportion where M2 >= M1 (higher better for pearson)
    p_value_kendall = np.mean(deltas_kendall <= 0) # Proportion where M2 >= M1 (higher better for kendall's tau)
    p_value_rmse = np.mean(deltas_rmse >= 0) # Proprotion where M2 has lower RMSE (better) than M1
    
    return_dict = {
        "p_value_pearson": p_value_pearson,
        "mean_delta_pearson": np.mean(deltas_pearson),
        "95_confint_pearson": np.percentile(deltas_pearson, [2.5, 97.5]),
        "p_value_kendall": p_value_kendall,
        "mean_delta_kendall": np.mean(deltas_kendall),
        "95_confint_kendall": np.percentile(deltas_kendall, [2.5, 97.5]),
        "p_value_rmse": p_value_rmse,
        "mean_delta_rmse": np.mean(deltas_rmse),
        "95_confint_rmse": np.percentile(deltas_rmse, [2.5, 97.5]), 
    } 
    
    return return_dict 
def main():
    args = parse_args()

    
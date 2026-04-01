"""
compute_bootstrap.py
=======================

This script will allow for the computing of bootstrap confidence intervals for model performance metrics (PCC, Kendall's Tau) 
as well for the computation of p-values to test for a significant difference between two models performance metric.

Like other scripts we are going to pass a config with all the required paramaters to allow us to process in larger batches.
Structure the configs as follows:
output_confint: "bootstrap_stats_confint.csv"
output_comparission: "bootstrap_stats_comparision.csv"

experiments:
- model_name_1: TAGCN-Drop-0-2-K-3-L-5-Dim-256-Intra-Zero-Ligand-Bias
  timestamp_1: '20260321_161700'
  model_name_2: TAGCN-Drop-0-2-K-3-L-5-Dim-256-Inter-Zero-Ligand-Bias
  timestamp_2: '20260321_161700'
  confidence_interval_1: true
  confidence_interval_2: true
  comparison: true 
  metrics: ["RMSE", "Pearson", "Kendall"]
  seed: 37  
  n_bootstraps: 10000

Usage:
    python compute_bootstrap.py --config_path config/bootstrap_config.yml
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import bootstrap
from scipy.stats import pearsonr, kendalltau
import sys
from tqdm import tqdm
import yaml

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

# Get predictions dir
predictions_dir = project_root / "output" / "predictions"
bootstrap_dir = project_root / "output" / "bootstrap"

predictions_dir.mkdir(parents=True, exist_ok=True)
bootstrap_dir.mkdir(parents=True, exist_ok=True)

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

def pearson(x, y):
    return pearsonr(x, y)[0]

def kendall(x, y):
    return kendalltau(x, y)[0]

def rmse(x, y):
    return np.sqrt(np.mean((x-y)**2))

def bootstrap_confint(preds, targets, seed, n_bootstraps=10000):
    """ compute bootstrap confidence intervals for all three performance metrics 
    """
    data = (preds, targets)

    pearsons_bca = bootstrap(data=data, statistic=pearson, confidence_level=0.95, method="BCa", paired=True, random_state=seed, n_resamples=n_bootstraps)
    kendalls_bca = bootstrap(data=data, statistic=kendall, confidence_level=0.95, method="BCa", paired=True, random_state=seed, n_resamples=n_bootstraps)
    rmse_bca = bootstrap(data=data, statistic=rmse, confidence_level=0.95, method="BCa", paired=True, random_state=seed, n_resamples=n_bootstraps)
    

    # Also compute the pearson, kendalls tau and RMSE 
    rho_og = pearson(preds, targets)
    tau_og = kendall(preds, targets)
    rmse_og = rmse(preds, targets)    

    return_dict = {
        "95_confint_pearson": (pearsons_bca.confidence_interval.low, pearsons_bca.confidence_interval.high),
        "pearson_original": rho_og,
        "95_confint_kendalls": (kendalls_bca.confidence_interval.low, kendalls_bca.confidence_interval.high),
        "kendall_original": tau_og,
        "95_confint_rmses": (rmse_bca.confidence_interval.low, rmse_bca.confidence_interval.high),
        "rmse_original": rmse_og
    }

    return return_dict

def bootstrap_comparison(m1_preds, m2_preds, targets, seed, n_bootstraps=10000):
    """ Performs the computations for the hypothesis test found in: https://www.nature.com/articles/s42004-025-01428-y. Comparing
    the performance metrics of two models. Null hypothesis is that model 2 is better than model 1, we get significance if that is not the case. 
    """ 
    
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
        tau_m2 = kendalltau(m2_preds[idx], targets[idx])[0]
        rmse_m1 = np.sqrt(np.mean((m1_preds[idx] - targets[idx])**2))
        rmse_m2 = np.sqrt(np.mean((m2_preds[idx] - targets[idx])**2)) 
        
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


def bootstrap_confint(preds, targets, seed, n_bootstraps=10000):
    """ compute bootstrap confidence intervals for all three performance metrics 
    """
    data = (preds, targets)

    pearsons_bca = bootstrap(data=data, statistic=pearson, confidence_level=0.95, method="BCa", paired=True, random_state=seed, n_resamples=n_bootstraps)
    kendalls_bca = bootstrap(data=data, statistic=kendall, confidence_level=0.95, method="BCa", paired=True, random_state=seed, n_resamples=n_bootstraps)
    rmse_bca = bootstrap(data=data, statistic=rmse, confidence_level=0.95, method="BCa", paired=True, random_state=seed, n_resamples=n_bootstraps)
    

    # Also compute the pearson, kendalls tau and RMSE 
    rho_og = pearson(preds, targets)
    tau_og = kendall(preds, targets)
    rmse_og = rmse(preds, targets)    

    return_dict = {
        "95_confint_pearson": (pearsons_bca.confidence_interval.low, pearsons_bca.confidence_interval.high),
        "pearson_original": rho_og,
        "95_confint_kendalls": (kendalls_bca.confidence_interval.low, kendalls_bca.confidence_interval.high),
        "kendall_original": tau_og,
        "95_confint_rmses": (rmse_bca.confidence_interval.low, rmse_bca.confidence_interval.high),
        "rmse_original": rmse_og
    }

    return return_dict
    
def main():
    args = parse_args()
    config_path = args.config_path

    # Turn config_path into absolute path
    config_path = project_root / config_path
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    
    # Store a list of dicts for the confidence intervals and comparissions
    confidence_intervals = []
    comparisons = []
    for experiment in tqdm(config["experiments"], desc="Running bootstrap computations", file=sys.stdout):
        seed = experiment["seed"]
        n_bootstraps = experiment["n_bootstraps"]

        # Get the name and timestamp so we can load the predictions file for this experiment
        model_name_1 = experiment["model_name_1"]
        timestamp_1 = experiment["timestamp_1"]

        model_1_file_name = timestamp_1 + "_" + model_name_1 + "_predictions.csv"
        model_1_file_path = (predictions_dir / model_1_file_name).resolve()

        df_model_1 = pd.read_csv(model_1_file_path)

        model_1_preds = df_model_1["test_preds_ensemble"].to_numpy()
        targets = df_model_1["test_targets"].to_numpy()

        model_name_2 = experiment["model_name_2"]
        timestamp_2 = experiment["timestamp_2"]

        if model_name_2 != "" and timestamp_2 != "":
            model_2_file_name = timestamp_2 + "_" + model_name_2 + "_predictions.csv"
            model_2_file_path = (predictions_dir / model_2_file_name).resolve()



            df_model_2 = pd.read_csv(model_2_file_path)

            model_2_preds = df_model_2["test_preds_ensemble"].to_numpy()
        else:
            df_model_2 = None # just so it is defined 

        # Execute the bootstrap computations according to the config
        if experiment["confidence_interval_1"]:
            model_1_confint = bootstrap_confint(preds=model_1_preds, targets=targets, seed=seed, n_bootstraps=n_bootstraps)

            tmp_dict = {
                "timestamp": timestamp_1,
                "experiment_name": model_name_1,
                "pearson_original": model_1_confint["pearson_original"],
                "pearson_2_5th_percentile": model_1_confint["95_confint_pearson"][0],
                "pearson_97_5th_percentile": model_1_confint["95_confint_pearson"][1],
                "kendall_original": model_1_confint["kendall_original"],  
                "kendall_2_5th_percentile": model_1_confint["95_confint_kendalls"][0],
                "kendall_97_5th_percentile": model_1_confint["95_confint_kendalls"][1],
                "rmse_original": model_1_confint["rmse_original"], 
                "rmse_2_5th_percentile": model_1_confint["95_confint_rmses"][0],
                "rmse_97_5th_percentile": model_1_confint["95_confint_rmses"][1]  
            }
            confidence_intervals.append(tmp_dict)

        if experiment["confidence_interval_2"]:
            model_2_confint = bootstrap_confint(preds=model_2_preds, targets=targets, seed=seed, n_bootstraps=n_bootstraps)

            tmp_dict = {
                "timestamp": timestamp_2,
                "experiment_name": model_name_2,
                "pearson_original": model_2_confint["pearson_original"],
                "pearson_2_5th_percentile": model_2_confint["95_confint_pearson"][0],
                "pearson_97_5th_percentile": model_2_confint["95_confint_pearson"][1],
                "kendall_original": model_2_confint["kendall_original"],   
                "kendall_2_5th_percentile": model_2_confint["95_confint_kendalls"][0],
                "kendall_97_5th_percentile": model_2_confint["95_confint_kendalls"][1],
                "rmse_original": model_2_confint["rmse_original"],  
                "rmse_2_5th_percentile": model_2_confint["95_confint_rmses"][0],
                "rmse_97_5th_percentile": model_2_confint["95_confint_rmses"][1]  
            }
            confidence_intervals.append(tmp_dict)


        if experiment["comparison"]:
            comparison_results = bootstrap_comparison(m1_preds=model_1_preds, m2_preds=model_2_preds, targets=targets, seed=seed, n_bootstraps=n_bootstraps)
            tmp_dict = {
                "timestamp_1": timestamp_1,
                "experiment_name_1": model_name_1,
                "timestamp_2": timestamp_2,
                "experiment_name_2": model_name_2,
                "p_value_pearson": comparison_results["p_value_pearson"],
                "mean_delta_pearson": comparison_results["mean_delta_pearson"],
                "pearson_2_5th_percentile": comparison_results["95_confint_pearson"][0],
                "pearson_97_5th_percentile": comparison_results["95_confint_pearson"][1],
                "p_value_kendall": comparison_results["p_value_kendall"],
                "mean_delta_kendall": comparison_results["mean_delta_kendall"],
                "kendall_2_5th_percentile": comparison_results["95_confint_kendall"][0],
                "kendall_97_5th_percentile": comparison_results["95_confint_kendall"][1],
                "p_value_rmse": comparison_results["p_value_rmse"],
                "mean_delta_rmse": comparison_results["mean_delta_rmse"],
                "rmse_2_5th_percentile": comparison_results["95_confint_rmse"][0],
                "rmse_97_5th_percentile": comparison_results["95_confint_rmse"][1]  
            } 
            comparisons.append(tmp_dict)

    # Save results
    if confidence_intervals != []:

        output_path = (bootstrap_dir / config["output_confint"]).resolve()

        df = pd.DataFrame(confidence_intervals)
        df.to_csv(output_path, mode="a", index=False, header=not output_path.exists())
    
    if comparisons != []:

        output_path = (bootstrap_dir / config["output_comparission"]).resolve()

        df = pd.DataFrame(comparisons)
        df.to_csv(output_path, mode="a", index=False, header=not output_path.exists())

if __name__ == "__main__":
    main()




    
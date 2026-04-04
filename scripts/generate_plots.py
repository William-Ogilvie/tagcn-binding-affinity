"""
generate_plots.py
=================

Script to generate comparative plots (Bubble plots, Loss/Metric curves) from training statistics.

Usage:
    python scripts/generate_plots.py --config_path config/plotting_config.yml
"""


import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, kendalltau

# Get the absolute path of the current script
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from training statistics.")
    parser.add_argument("--config_path", type=str, required=True, help="Relative path to the plotting config file.") 
    return parser.parse_args()

def plot_bubble(experiments_config, stats_dir, output_dir, file_name_tag, type_x, type_x_name, type_y, type_y_name, training_time, title, label_points=True, alternate_labels=False, bootstrap_bars=False):
    """
    Generates a bubble plot comparing experiments, does one both with training_time for size and another with a fixed size.
    X-axis: type_x (function parameter)
    Y-axis: type_Y (function parameter)    
    """
    print("Generating Bubble Plot...")
    stats_path = stats_dir / "training_stats.csv"
    
    if not stats_path.exists():
        print(f"Warning: {stats_path} not found. Skipping bubble plot.")
        return

    df = pd.read_csv(stats_path)
    
    # Filter for experiments in the config
    target_experiments = {(exp["name"], str(exp["time_stamp"])) for exp in experiments_config}
    
    # Ensure timestamp is string for matching
    df["timestamp"] = df["timestamp"].astype(str)
    
    # Filter dataframe
    # We match on both name and timestamp to be precise
    mask = df.apply(lambda x: (x["experiment_name"], str(x["timestamp"])) in target_experiments, axis=1)
    plot_df = df[mask].copy()

    # Map experiment_name and timestamp to plot_labels
    label_map = {}
    for exp in experiments_config:
        name = exp["name"]
        ts = str(exp["time_stamp"])
        label_map[(name, ts)] = exp.get("alternate_label", name) if alternate_labels else name

    plot_df["plot_labels"] = plot_df.apply(
        lambda x: label_map.get((x["experiment_name"], str(x["timestamp"])), x["experiment_name"]), 
        axis=1
    )

    if bootstrap_bars:
        bootstrap_path = project_root / "output" / "bootstrap" / "bootstrap_stats_confint.csv"
        if bootstrap_path.exists():
            boot_df = pd.read_csv(bootstrap_path)
            boot_df["timestamp"] = boot_df["timestamp"].astype(str)
            plot_df = plot_df.merge(boot_df, on=["experiment_name", "timestamp"], how="left")
        else:
            print(f"Warning: {bootstrap_path} not found. Skipping bootstrap bars.")
            bootstrap_bars = False

    if plot_df.empty:
        print("No matching experiments found in training_stats.csv for bubble plot.")
        return
   
    
    # If the bool training_time is true then the bubble sizes are scaled by training_time (the reason we may not want this is on the cluster if you use an array job then it is almost random what hardware you get)
    if training_time:
        # Plotting with training time for size
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Create bubble plot 
        scatter = sns.scatterplot(
            data=plot_df,
            x=type_x,
            y=type_y,
            hue="plot_labels", 
            size="training_time_seconds",
            sizes=(100, 1000),
            alpha=0.7,
            palette="viridis"
        )

        if bootstrap_bars:
            metric_map = {"test_set_pearson": "pearson", "test_set_kendall": "kendall", "test_set_rmse": "rmse"}
            x_prefix = metric_map.get(type_x)
            y_prefix = metric_map.get(type_y)
            
            xerr, yerr = None, None
            if x_prefix and f"{x_prefix}_2_5th_percentile" in plot_df.columns:
                x_err_low = np.maximum(0, plot_df[type_x] - plot_df[f"{x_prefix}_2_5th_percentile"].fillna(plot_df[type_x]))
                x_err_high = np.maximum(0, plot_df[f"{x_prefix}_97_5th_percentile"].fillna(plot_df[type_x]) - plot_df[type_x])
                xerr = [x_err_low, x_err_high]
            if y_prefix and f"{y_prefix}_2_5th_percentile" in plot_df.columns:
                y_err_low = np.maximum(0, plot_df[type_y] - plot_df[f"{y_prefix}_2_5th_percentile"].fillna(plot_df[type_y]))
                y_err_high = np.maximum(0, plot_df[f"{y_prefix}_97_5th_percentile"].fillna(plot_df[type_y]) - plot_df[type_y])
                yerr = [y_err_low, y_err_high]
                
            plt.errorbar(
                x=plot_df[type_x], y=plot_df[type_y],
                xerr=xerr, yerr=yerr,
                fmt='none', ecolor='gray', alpha=0.5, capsize=3, zorder=0
            )

        # Label points
        if label_points:
            labels_list = [] 
            for line in range(0, plot_df.shape[0]):
                labels_list.append(plt.text(
                    plot_df.iloc[line][type_x], 
                    plot_df.iloc[line][type_y], 
                    plot_df.iloc[line]["plot_labels"],
                    plot_df.iloc[line]["training_time_seconds"], 
                    horizontalalignment='center', 
                    size='small', 
                    color='black', 
                    weight='semibold'
                ))
            adjust_text(labels_list, 
                        only_move={'points':'y', 'texts':'xy'}, # Focus on moving text
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
            
            
        plt.rcParams.update({'font.size': 20}) 
        plt.rcParams.update({'axes.labelsize': 22})
        plt.rcParams.update({'xtick.labelsize': 18})
        #plt.title(title, fontsize=22)
        plt.xlabel(f"{type_x_name}", fontsize=22)
        plt.ylabel(f"{type_y_name}", fontsize=22)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend(loc='lower left')
        plt.tight_layout()
        
        plt.savefig(output_dir / f"{file_name_tag}_comparison_bubble_plot_{type_y}_against_{type_x}_with_training_time.pdf")
        plt.close()
        print("Bubble plot saved with training time.")
    else:
        # Plotting without training time for size
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Create bubble plot 
        scatter = sns.scatterplot(
            data=plot_df,
            x=type_x,
            y=type_y,
            hue="plot_labels",
            s = 400, 
            alpha=0.7,
            palette="viridis"
        )

        if bootstrap_bars:
            metric_map = {"test_set_pearson": "pearson", "test_set_kendall": "kendall", "test_set_rmse": "rmse"}
            x_prefix = metric_map.get(type_x)
            y_prefix = metric_map.get(type_y)
            
            xerr, yerr = None, None
            if x_prefix and f"{x_prefix}_2_5th_percentile" in plot_df.columns:
                x_err_low = np.maximum(0, plot_df[type_x] - plot_df[f"{x_prefix}_2_5th_percentile"].fillna(plot_df[type_x]))
                x_err_high = np.maximum(0, plot_df[f"{x_prefix}_97_5th_percentile"].fillna(plot_df[type_x]) - plot_df[type_x])
                xerr = [x_err_low, x_err_high]
            if y_prefix and f"{y_prefix}_2_5th_percentile" in plot_df.columns:
                y_err_low = np.maximum(0, plot_df[type_y] - plot_df[f"{y_prefix}_2_5th_percentile"].fillna(plot_df[type_y]))
                y_err_high = np.maximum(0, plot_df[f"{y_prefix}_97_5th_percentile"].fillna(plot_df[type_y]) - plot_df[type_y])
                yerr = [y_err_low, y_err_high]
                
            plt.errorbar(
                x=plot_df[type_x], y=plot_df[type_y],
                xerr=xerr, yerr=yerr,
                fmt='none', ecolor='gray', alpha=0.5, capsize=3, zorder=0
            )

        # Label points
        if label_points:
            labels_list = []
            for line in range(0, plot_df.shape[0]):
                labels_list.append(plt.text(
                    plot_df.iloc[line][type_x], 
                    plot_df.iloc[line][type_y], 
                    plot_df.iloc[line]["plot_labels"], 
                    horizontalalignment='center', 
                    size='small', 
                    color='black', 
                    weight='semibold'
                ))

            adjust_text(labels_list, 
                        only_move={'points':'y', 'texts':'xy'}, # Focus on moving text
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
            
            
            
        plt.rcParams.update({'font.size': 20}) 
        plt.rcParams.update({'axes.labelsize': 22})
        plt.rcParams.update({'xtick.labelsize': 18})
        
        # plt.title(title, fontsize=22)
        plt.xlabel(f"{type_x_name}", fontsize=22)
        plt.ylabel(f"{type_y_name}", fontsize=22)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        plt.savefig(output_dir / f"{file_name_tag}_comparison_bubble_plot_{type_y}_against_{type_x}_without_training_time.pdf")
        plt.close()

    

def plot_metrics_over_epochs(experiments_config, stats_dir, output_dir, file_name_tag):
    """
    Generates line plots for Validation Pearson and Loss over epochs.
    Aggregates seeds to show Mean +/- Std (Representative Member).
    """
    print("Generating Metrics over Epochs Plots...")
    
    all_data = []

    for exp in experiments_config:
        name = exp["name"]
        timestamp = str(exp["time_stamp"])
        label = exp.get("label", name) # Use label if provided, else name
        
        file_path = stats_dir / f"{timestamp}_{name}_epoch_stats.csv"
        
        if not file_path.exists():
            print(f"Warning: Stats file for {name} ({timestamp}) not found at {file_path}. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        df["experiment_label"] = label
        all_data.append(df)

    if not all_data:
        print("No data found for metric plots.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Define metrics to plot
    metrics = {
        "val_pearson": "Validation Pearson Correlation",
        "val_loss": "Validation Loss (MSE)",
        "val_kendall": "Validation Kendall Correlation"
    }

    for metric, title in metrics.items():
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Seaborn's lineplot automatically calculates mean and confidence interval (95% CI by default)
        # across the 'seed' values for each epoch.
        sns.lineplot(data=combined_df, x="epoch", y=metric, hue="experiment_label")
        
        plt.title(f"Comparison: {title}", fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.legend(title="Experiment")
        plt.tight_layout()
        
        plt.savefig(output_dir / f"{file_name_tag}_comparison_{metric}.pdf")
        plt.close()
    
    print("Metric plots saved.")

def plot_true_vs_predicted(experiments_config, predictions_dir, output_dir):
    """
    Generates True vs Predicted scatter plots for each experiment.
    Includes y=x line and metrics (Pearson, RMSE, Kendall) in the title.
    """
    print("Generating True vs Predicted Plots...")
    
    for exp in experiments_config:
        name = exp["name"]
        timestamp = str(exp["time_stamp"])
        
        # Construct expected filename based on training_script.py output
        file_name = f"{timestamp}_{name}_predictions.csv"
        file_path = predictions_dir / file_name
        
        if not file_path.exists():
            print(f"Warning: Predictions file for {name} ({timestamp}) not found at {file_path}. Skipping.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Get required columns 
        true_col = "test_targets" 
        pred_col = "test_preds_ensemble" 
        
             
        y_pred = df[pred_col].values
        y_true = df[true_col].values
        
        # Compute metrics
        pearson_corr, _ = pearsonr(y_true, y_pred)
        kendall_corr, _ = kendalltau(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label='Predictions')
        
        # y=x line
        # Get the current limits of the plot after the scatter points are drawn
        lims = [
            np.min([plt.xlim(), plt.ylim()]),  # min of both axes
            np.max([plt.xlim(), plt.ylim()]),  # max of both axes
        ]

        # Plot the line using those limits  
        plt.plot(lims, lims, color='red', linestyle='--', alpha=0.5, zorder=0, label='y=x')

        # Force the aspect ratio to be equal so the line is actually 45 degrees
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(lims)
        plt.ylim(lims)

        # Old y=x
        # min_val = min(np.min(y_true), np.min(y_pred))
        # max_val = max(np.max(y_true), np.max(y_pred))
        # margin = (max_val - min_val) * 0.05
        # plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
        #          color='red', linestyle='--', label='y=x')

        plt.xlabel('True Binding Affinity (pK)')
        plt.ylabel('Predicted Binding Affinity')
        plt.title(f'{name}\nPearson: {pearson_corr:.3f}, RMSE: {rmse:.3f}, Kendall: {kendall_corr:.3f}')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plot_filename = f"{timestamp}_{name}_true_vs_pred.pdf"
        plt.savefig(output_dir / plot_filename)
        plt.close()
        print(f"Saved plot to {output_dir / plot_filename}")

def main():
    args = parse_args()
    config_path = project_root / args.config_path
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = project_root / config["output_dir"]
    stats_dir = project_root / config["stats_dir"]
    predictions_dir = project_root / config.get("predictions_dir", "output/predictions")
    file_name_tag = config["file_name_tag"]    

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which plots to generate
    if config["plots"].get("bubble", False):
        label_points = config["plots"].get("bubble_labels", True)
        alternate_labels = config["plots"].get("alternate_labels", False)
        bootstrap_bars = config["plots"].get("bootstrap_bars", False)
        title = config["plots"].get("title", "")
        plot_bubble(experiments_config=config["experiments"], stats_dir=stats_dir, output_dir=output_dir, file_name_tag=file_name_tag, type_y="test_set_rmse", type_y_name="RMSE", type_x="test_set_pearson", type_x_name="TPCC", training_time=False, title=title, label_points=label_points, alternate_labels=alternate_labels, bootstrap_bars=bootstrap_bars)
        plot_bubble(experiments_config=config["experiments"], stats_dir=stats_dir, output_dir=output_dir, file_name_tag=file_name_tag, type_y="test_set_rmse", type_y_name="RMSE", type_x="test_set_kendall", type_x_name=r"K$\tau$", training_time=False, title=title, label_points=label_points, alternate_labels=alternate_labels, bootstrap_bars=bootstrap_bars)
        plot_bubble(experiments_config=config["experiments"], stats_dir=stats_dir, output_dir=output_dir, file_name_tag=file_name_tag, type_x="test_set_pearson", type_x_name="PCC", type_y="test_set_kendall", type_y_name=r"K$\tau$", training_time=False, title=title, label_points=label_points, alternate_labels=alternate_labels, bootstrap_bars=bootstrap_bars)
        
        
    if config["plots"].get("metrics_over_epochs", False):
        plot_metrics_over_epochs(config["experiments"], stats_dir, output_dir, file_name_tag)
        
    if config["plots"].get("true_vs_predicted", False):
        plot_true_vs_predicted(config["experiments"], predictions_dir, output_dir)

if __name__ == "__main__":
    main()
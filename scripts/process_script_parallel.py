"""
process_script_parallel.py
============================

This script will process the models generated from the training_script_parallel.py, ensuring we generate predictions on the test set for the ensemble.
We use a similar config to that when training the models, the key difference is we need to ensure each experiment has the correct run_id inside its config
so we load the correct files. We have an example config in config/process_config.yml

Usage:
    python scripts/process_script_parallel.py --config_path config/process_config.yml --device auto
"""
import argparse
import torch
import time
from pathlib import Path
import yaml
from tagcn_bind import Trainer, GATv2Net, GATv2Net_v2, TAGCNet, TAGCNet_v2, TAGCNet_v3, init_weights, PDBDataset
from torch_geometric.loader import DataLoader
from scipy.optimize import minimize
from scipy.stats import pearsonr, kendalltau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

def initalise_model(model_name, node_feat_dim, edge_feat_dim, hidden_dim, num_GNN_layers, activation, head=3, K=3, dropout_rate=0.0):

    # Generate a config dict:
    config_dict = {
        "num_gnn_layers": num_GNN_layers,
        "head": head,
        "K": K,
        "hidden_dim": hidden_dim,
        "activation": activation,
        "dropout_rate": dropout_rate,
        "training": True
    }

    if model_name == "GATv2":
        model = GATv2Net(node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim, config=config_dict)
    elif model_name == "TAGCN":
        model = TAGCNet(node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim, config=config_dict)
    elif model_name == "GATv2_v2": 
        model = GATv2Net_v2(node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim, config=config_dict)
    elif model_name == "TAGCN_v2":
        model = TAGCNet_v2(node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim, config=config_dict)
    elif model_name == "TAGCN_v3":
        model = TAGCNet_v3(node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim, config=config_dict)
 
    else:
        raise ValueError(f"Coulnd't identify model name: {model_name}")
    
    return model

def find_feat_edge_dim(train_graphs_dir):

    # Load the first graph in training and get the dimension of the node and edge features
    # Remember graphs are stored as shards 
    shard_files = list(train_graphs_dir.glob("*.pt"))
    try: 
        first_shard_path = shard_files[0]
        first_shard = torch.load(first_shard_path, weights_only=False)
    except Exception as e:
        raise ValueError(f"ERROR: Failed to inspect shard: {e}")  
    
    first_id = list(first_shard.keys())[0]
    uid, graph, pK = first_shard[first_id]

    feat_dim = len(graph[1][0])

    edge_dim = len(graph[3][0])

    return feat_dim, edge_dim

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GNN model for binding affinity prediction.")

    # Config path
    parser.add_argument("--config_path", type = str, default="config/experiments_config.yml", help="Relative file path to config with experiments.")    # Model selection

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu).")

    return parser.parse_args()

def get_device(device_str):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str

def main():
    start_time = time.time()
    args = parse_args()
    device_str = get_device(args.device)
    config_path = args.config_path

    # Turn config_path into absolute path
    config_path = project_root / config_path
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Loop through the experiments in the config
    for experiment in tqdm(config["experiments"], desc="Processing Experiments", file=sys.stdout): 
    

        # Unpack the experiment dict 
        experiment_name = experiment["name"]
        run_id = experiment["run_id"]
        model_name = experiment["args"]["model"] 
        epochs = experiment["args"]["epochs"]
        batch_size = experiment["args"]["batch_size"]
        lr = experiment["args"]["lr"]
        early_stopping = experiment["args"]["early_stopping"]
        early_stopping_metric = experiment["args"]["early_stopping_metric"] 
        hidden_dim = experiment["args"]["hidden_dim"]
        head = experiment["args"]["head"]
        K_param = experiment["args"]["K"]
        dropout_rate = experiment["args"]["dropout_rate"]
        num_GNN_layers = experiment["args"]["num_GNN_layers"]
        activation_function = experiment["args"]["activation_function"]
        weight_decay = experiment["args"]["weight_decay"] 
        seeds = experiment["args"]["seeds"]
        scaling_stats_pt = experiment["args"]["scaling_stats_pt"] 
        train_data_csv = experiment["args"]["train_data_csv"]
        val_data_csv = experiment["args"]["val_data_csv"]
        test_data_csv = experiment["args"]["test_data_csv"]  
        train_graphs_dir = experiment["args"]["train_graphs_dir"] 
        val_graphs_dir = experiment["args"]["val_graphs_dir"]
        test_graphs_dir = experiment["args"]["test_graphs_dir"]
        model_save_dir = experiment["args"]["model_save_dir"]
        predictions_save_dir = experiment["args"]["predictions_save_dir"]  
        save_training_stats_dir = experiment["args"]["save_training_stats_dir"]
        plots_dir = experiment["args"]["plots_dir"]
        scaling_stats_path = project_root / scaling_stats_pt

        # As we added the standardisation options later we use .get so they default to True for earlier models
        standardise_aevs = experiment["args"].get("standardise_aevs", True)
        standardise_targets = experiment["args"].get("standardise_targets", True)



        # Setup output folders
        model_save_dir = project_root / model_save_dir
        predictions_save_dir = project_root / predictions_save_dir
        save_training_stats_dir = project_root / save_training_stats_dir
        plots_dir = project_root / plots_dir
        

        model_save_dir.mkdir(parents = True, exist_ok = True)
        predictions_save_dir.mkdir(parents = True, exist_ok = True)
        save_training_stats_dir.mkdir(parents = True, exist_ok = True)
        plots_dir.mkdir(parents = True, exist_ok = True)

        train_dataset_name = train_data_csv.split("/")[-1][:-4]
        val_dataset_name = val_data_csv.split("/")[-1][:-4]
        test_dataset_name = test_data_csv.split("/")[-1][:-4]

        test_data_csv_path = project_root / test_data_csv
            
    
        train_graphs_dir = project_root / train_graphs_dir
        val_graphs_dir = project_root / val_graphs_dir
        test_graphs_dir = project_root / test_graphs_dir
        test_set = PDBDataset(data_dir=test_graphs_dir, dataset_name=test_dataset_name)
  

        # Get the node and edge feature dimensions
        node_feat_dim, edge_feat_dim = find_feat_edge_dim(train_graphs_dir=train_graphs_dir)

        # Get the model, we just initalise an empty model in the same way as the training script that way the Trainer class is initalised correctly. We will use the weights storred in output/models for actual inference 
        model = initalise_model(model_name=model_name, edge_feat_dim=edge_feat_dim, node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, head=head, num_GNN_layers=num_GNN_layers, activation=activation_function, K=K_param, dropout_rate=dropout_rate)

        # Ensure model is in double precision to match the dataset
        model = model.double()

        # Initalise weights
        model.apply(init_weights)

        # Initalise device
        if device_str == "cuda":
            print("GPU is available")
            device = torch.device("cuda")
        else:
            print("Falling back to cpu")
            device = torch.device("cpu")

        # Initialise the trainer 
        trainer = Trainer(model=model, device=device, stats_path=scaling_stats_path)

            
        # ---------------------------------------------------------
        # Prediction 
        # ---------------------------------------------------------

        print(f"Getting predictions for {experiment_name} ensemble...")
        # Loaders for inference (no shuffle)
        test_loader_inf = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers = 0, # for test set isn't much point in using multiple workers due to size of data set
            pin_memory = True
        )

            
        
        # Iterate through trained models and compute predictions on test set
        ensemble_test_preds = []
        test_targets = []
        test_df_dict = {}
        all_models_exist = True
        for seed in seeds: 
            model_path = model_save_dir / f"{run_id}_{experiment_name}_model_{seed}.pt"
                    
            # Load model weights into the existing trainer's model
            try:
                trainer.model.load_state_dict(torch.load(model_path, weights_only=False))
            except:
                print(f"Failed to load model: {model_path}")
                all_models_exist = False
                break

            # If the model uses dropout it needs to be turned off here:
            if model_name == "GATv2_v2" or model_name == "TAGCN_v2":
                trainer.model.training = False
            
                
            # Predict Test
            test_preds, test_targets = trainer.predict(test_loader_inf, standardise_aevs=standardise_aevs, standardise_targets=standardise_targets)
            ensemble_test_preds.append(test_preds)

            # Add predictions to test_df_dict
            test_df_dict[f"test_preds_seed_{seed}"] = test_preds 

        # If we failed to load one of the ensemble memebers we are going to have to skip this
        if not all_models_exist:
            print(f"Missing models for {experiment_name}, skipping for now")
            continue

        # Calculate Ensemble Test Metrics 
        ensemble_test_preds_stacked = np.stack(ensemble_test_preds) # shape: (10, N)
        ensemble_test_preds_final = np.mean(ensemble_test_preds_stacked, axis=0) 
        test_rmse = np.sqrt(np.mean((test_targets - ensemble_test_preds_final)**2))
        test_pearson, _ = pearsonr(test_targets, ensemble_test_preds_final)
        test_kendall, _ = kendalltau(test_targets, ensemble_test_preds_final)

        # Finish the dict so we can create and save it as a csv
        test_df_dict["test_preds_ensemble"] = ensemble_test_preds_final
        test_df_dict["test_targets"] = test_targets

        test_df = pd.DataFrame(test_df_dict)

        test_df.to_csv(predictions_save_dir / f"{run_id}_{experiment_name}_predictions.csv", index = False)

        # ---------------------------------------------------------
        # Plotting and Stats Saving
        # ---------------------------------------------------------

        print(f"Finished predictions, starting epoch stats plots for {experiment_name}...")
        # Below is a copy of the plotting code from training.py, we are going to loop through all the saved rough epoch stats to recover the required information for the plots

        # Loop through the seeds and recover epoch stats data
        all_seeds_train_loss = []
        all_seeds_val_loss = []
        all_seeds_train_pearson = []
        all_seeds_val_pearson = []
        all_seeds_train_kendall = []
        all_seeds_val_kendall = []
        for seed in seeds:
            epoch_stats_path = save_training_stats_dir / "rough" / f"{run_id}_{experiment_name}_{seed}_epoch_stats.csv"
            # Load the epoch stats csv
            try:
                df = pd.read_csv(filepath_or_buffer=epoch_stats_path) 
            except:
                print(f"Failed to read {epoch_stats_path} skipping.")
                continue
            all_seeds_train_loss.append(df["train_loss"])
            all_seeds_val_loss.append(df["val_loss"])
            all_seeds_train_pearson.append(df["train_pearson"])
            all_seeds_val_pearson.append(df["val_pearson"])
            all_seeds_train_kendall.append(df["train_kendall"])
            all_seeds_val_kendall.append(df["val_kendall"])
            

        # Plot Loss
        if all_seeds_train_loss:
            plt.figure(figsize=(10, 6))
                
            for i, (train_loss, val_loss) in enumerate(zip(all_seeds_train_loss, all_seeds_val_loss)):
                epochs_range = range(1, len(train_loss) + 1)
                # Only label the first trace to keep legend clean
                lbl_train = 'Train Loss' if i == 0 else None
                lbl_val = 'Validation Loss' if i == 0 else None
                    
                plt.plot(epochs_range, train_loss, color='blue', alpha=0.3, label=lbl_train)
                plt.plot(epochs_range, val_loss, color='orange', alpha=0.3, label=lbl_val)

            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title(f'Training and Validation Loss (All Seeds): {experiment_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / f"{experiment_name}_{run_id}_loss.pdf")
            plt.close()

            # Plot Pearson
            plt.figure(figsize=(10, 6))
            for i, (train_pearson, val_pearson) in enumerate(zip(all_seeds_train_pearson, all_seeds_val_pearson)):
                epochs_range = range(1, len(val_pearson) + 1)
                lbl_train = "Train Pearson" if i == 0 else None  
                lbl_val = 'Validation Pearson' if i == 0 else None
                plt.plot(epochs_range, train_pearson, color = "blue", alpha = 0.3, label=lbl_train)
                plt.plot(epochs_range, val_pearson, color='green', alpha=0.3, label=lbl_val)

            plt.xlabel('Epoch')
            plt.ylabel('Pearson Correlation')
            plt.title(f'Training and Validation Pearson Correlation (All Seeds): {experiment_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / f"{experiment_name}_{run_id}_pearson.pdf")
            plt.close()

            # Plot Kendall
            plt.figure(figsize=(10, 6))
            for i, (train_kendall, val_kendall) in enumerate(zip(all_seeds_train_kendall, all_seeds_val_kendall)):
                epochs_range = range(1, len(val_kendall) + 1)
                lbl_train = 'Train Kendall' if i == 0 else None
                lbl_val = 'Validation Kendall' if i == 0 else None
                plt.plot(epochs_range, train_kendall, color='blue', alpha=0.3, label=lbl_train)
                plt.plot(epochs_range, val_kendall, color='purple', alpha=0.3, label=lbl_val)

            plt.xlabel('Epoch')
            plt.ylabel('Kendall Correlation')
            plt.title(f'Validation Kendall Correlation (All Seeds): {experiment_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / f"{experiment_name}_{run_id}_kendall.pdf")
            plt.close()

        # Save Per-Epoch Stats for External Plotting
        epoch_stats_rows = []
        for seed_idx, seed in enumerate(seeds):
            # Check if we have data for this seed
            if seed_idx < len(all_seeds_train_loss):
                t_loss = all_seeds_train_loss[seed_idx]
                v_loss = all_seeds_val_loss[seed_idx]
                t_pearson = all_seeds_train_pearson[seed_idx]
                v_pearson = all_seeds_val_pearson[seed_idx]
                t_kendall = all_seeds_train_kendall[seed_idx]
                v_kendall = all_seeds_val_kendall[seed_idx]
                    
                for ep in range(len(t_loss)):
                    epoch_stats_rows.append({
                        "epoch": ep + 1,
                        "seed": seed,
                        "train_loss": t_loss[ep],
                        "val_loss": v_loss[ep],
                        "train_pearson": t_pearson[ep],
                        "val_pearson": v_pearson[ep],
                        "train_kendall": t_kendall[ep],
                        "val_kendall": v_kendall[ep]
                    })
            
        pd.DataFrame(epoch_stats_rows).to_csv(save_training_stats_dir / f"{run_id}_{experiment_name}_epoch_stats.csv", index=False)

        # Final statistic to recover is the training time
        train_time = 0.0
        for seed in seeds:
            # Load the rough training_stats 
            training_stats_rough_path = save_training_stats_dir / "rough" / f"training_stats_{run_id}_{experiment_name}_{seed}.csv"

            try:
                df = pd.read_csv(training_stats_rough_path)
            except:
                print(f"Failed to read {training_stats_rough_path}, skipping")
                continue

            train_time_seed = df["training_time_seconds"][0]
            train_time += train_time_seed

        print(f"Finished epoch stats plots, saving training statistics for {experiment_name}...")
        # Save Stats
        stats_file = save_training_stats_dir / "training_stats.csv"
        stats_data = {
            "timestamp": run_id,
            "experiment_name": experiment_name,
            "test_set_rmse": test_rmse,
            "test_set_pearson": test_pearson,
            "test_set_kendall": test_kendall,
            "training_time_seconds": train_time
        }
        df_stats = pd.DataFrame([stats_data])
        if not stats_file.exists():
            df_stats.to_csv(stats_file, index=False)
        else:
            df_stats.to_csv(stats_file, mode='a', header=False, index=False)

        print(f"{experiment_name} has been fully processed!")

if __name__ == "__main__":
    main()




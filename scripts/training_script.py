"""
training_script.py
==================

Script to train a GNN model using TrainingManager.
Usage:
    python scripts/training_script.py --config_path config/experiments_config.yml --device auto 
"""
import argparse
import torch
import time
from pathlib import Path
import yaml
from tagcn_bind import Trainer, GATv2Net, TAGCNet, init_weights, PDBDataset
from torch_geometric.loader import DataLoader
from scipy.optimize import minimize
from scipy.stats import pearsonr, kendalltau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the absolute path of the current script
script_path = Path(__file__).resolve()

# Go up two levels to get the project root:
project_root = script_path.parent.parent

def initalise_model(model_name, node_feat_dim, edge_feat_dim, hidden_dim, num_GNN_layers, activation, head=3, K=3):

    # Generate a config dict:
    config_dict = {
        "num_gnn_layers": num_GNN_layers,
        "head": head,
        "K": 3,
        "hidden_dim": hidden_dim,
        "activation": activation
    }

    if model_name == "GATv2":
        model = GATv2Net(node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim, config=config_dict)
        return model
    elif model_name == "TAGCN":
        model = TAGCNet(node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim, config=config_dict)
    else:
        raise ValueError(f"Coulnd't identify model name: {model_name}")
    
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

    # Loop through the experiments:
    for experiment in config["experiments"]:
    
        # Timestamp for unique filenames
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Unpack the experiment dict
        experiment_name = experiment["name"]
        model_name = experiment["args"]["model"]
        epochs = experiment["args"]["epochs"]
        batch_size = experiment["args"]["batch_size"]
        lr = experiment["args"]["lr"]
        early_stopping = experiment["args"]["early_stopping"]
        early_stopping_metric = experiment["args"]["early_stopping_metric"] 
        hidden_dim = experiment["args"]["hidden_dim"]
        head = experiment["args"]["head"]
        K_param = experiment["args"]["K"]
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
        ensemble_metric = experiment["args"]["ensemble_metric"]
        save_training_stats_dir = experiment["args"]["save_training_stats_dir"]
        plots_dir = experiment["args"]["plots_dir"]
        scaling_stats_path = project_root / scaling_stats_pt

        # Setup output folders
        model_save_dir = project_root / model_save_dir
        predictions_save_dir = project_root / predictions_save_dir
        save_training_stats_dir = project_root / save_training_stats_dir
        plots_dir = project_root / plots_dir

        model_save_dir.mkdir(parents = True, exist_ok = True)
        predictions_save_dir.mkdir(parents = True, exist_ok = True)
        save_training_stats_dir.mkdir(parents = True, exist_ok = True)
        plots_dir.mkdir(parents = True, exist_ok = True)
 
        train_graphs_dir = project_root / train_graphs_dir
        val_graphs_dir = project_root / val_graphs_dir
        test_graphs_dir = project_root / test_graphs_dir
        train_set = PDBDataset(data_dir=train_graphs_dir)
        val_set = PDBDataset(data_dir=val_graphs_dir)
        test_set = PDBDataset(data_dir=test_graphs_dir)

        # Lists to store history for plotting across seeds
        all_seeds_train_loss = []
        all_seeds_val_loss = []
        all_seeds_val_pearson = []
        all_seeds_val_kendall = []

        # Loop through random seeds for ensemble
        for i, seed in enumerate(seeds):

            # Create DataLoaders, handles batching, shuffling and multiprocessing
            train_loader = DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers = 4, # num subprocessed used for data loading
                pin_memory = True, # copy tensors into CUDA before returning them
                prefetch_factor = 2 # number of batches loaded in advance by each worker (so they have something to do while training)
            )
            val_loader = DataLoader(
                dataset=val_set,
                batch_size=batch_size, 
                shuffle=False,
                num_workers = 2, # Validation set is only 10% of the size of the training set so need less workers
                pin_memory = True,
                prefetch_factor = 2
            )

            # Get the node and edge feature dimensions
            node_feat_dim, edge_feat_dim = find_feat_edge_dim(train_graphs_dir=train_graphs_dir)

            # Get the model 
            model = initalise_model(model_name=model_name, edge_feat_dim=edge_feat_dim, node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, head=head, num_GNN_layers=num_GNN_layers, activation=activation_function, K=K_param)

            # Initalise weights
            model.apply(init_weights)

            # Setup optimizer and loss
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.MSELoss() # loss function

            # Initalise device
            if device_str == "cuda":
                print("GPU is available")
                device = torch.device("cuda")
            else:
                print("Falling back to cpu")
                device = torch.device("cpu")

            # Initialise the trainer 
            trainer = Trainer(model=model, device=device, stats_path=scaling_stats_path)

            # Training loop with early stopping
            best_val_loss = float('inf')
            best_val_corr = float('-inf') # this is kinda overkill bc PC and Kendalls Tau >= -1, but have kept it in case change to a different metric later
            wait = 0

            print(f"Running training for {experiment_name}, model name {model_name}, seed {seed} of {i}/{len(seeds)}.")

            # Per seed history
            seed_train_loss = []
            seed_val_loss = []
            seed_val_pearson = []
            seed_val_kendall = []

            for epoch in range(epochs):

                train_loss, train_kendall_corr, train_pearson_corr = trainer.train_epoch(loader=train_loader, optimizer=optimizer, criterion=criterion)
                val_loss, val_kendall_corr, val_pearson_corr = trainer.validate(loader=val_loader, criterion=criterion)
                
                seed_train_loss.append(train_loss)
                seed_val_loss.append(val_loss)
                seed_val_pearson.append(val_pearson_corr)
                seed_val_kendall.append(val_kendall_corr)

                print(f"Epoch {epoch:03d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

                # Check for early stopping, we have a choise to stop on either mse (loss), kendalls tau or pearsons
                if early_stopping_metric == "mse":
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        torch.save(model.state_dict(), f"{model_save_dir}/{timestamp}_{experiment_name}_model_{i}.pt")
                    else:
                        wait += 1
                        if wait >= early_stopping:
                            print(f"Early stopping triggered at epoch {epoch}")
                            break
                elif early_stopping_metric == "kendall":
                    if val_kendall_corr > best_val_corr:
                        best_val_corr = val_kendall_corr
                        wait = 0
                        torch.save(model.state_dict(), f"{model_save_dir}/{timestamp}_{experiment_name}_model_{i}.pt")
                    else:
                        wait += 1
                        if wait >= early_stopping:
                            print(f"Early stopping triggered at epoch {epoch}")
                            break
                elif early_stopping_metric == "pearson":
                    if val_pearson_corr > best_val_corr:
                        best_val_corr = val_kendall_corr
                        wait = 0
                        torch.save(model.state_dict(), f"{model_save_dir}/{timestamp}_{experiment_name}_model_{i}.pt")
                    else:
                        wait += 1
                        if wait >= early_stopping:
                            print(f"Early stopping triggered at epoch {epoch}")
                            break
                else:
                    raise ValueError(f"Invalid early stopping metric {early_stopping_metric}")
    
        # ---------------------------------------------------------
        # Ensemble Optimization and Prediction
        # ---------------------------------------------------------
        print(f"Training finished for all seeds. Starting ensemble optimization ({ensemble_metric})...")

        train_time = time.time() - start_time      

        # Loaders for inference (no shuffle)
        val_loader_inf = DataLoader(
            dataset=val_set, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers = 2,
            pin_memory = True,
            prefetch_factor = 2
        )
        test_loader_inf = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers = 0, # for test set isn't much point in using multiple workers due to size of data set
            pin_memory = True,
            prefetch_factor = 0
        )
        
        val_preds_list = []
        test_preds_list = []
        val_targets = None
        test_targets = None
        
        # Iterate through trained models
        for i, seed in enumerate(seeds):
            model_path = model_save_dir / f"{timestamp}_{experiment_name}_model_{i}.pt"
            
            # Load model weights into the existing trainer's model
            trainer.model.load_state_dict(torch.load(model_path, weights_only=False))
            
            # Predict Validation
            v_preds, v_y = trainer.predict(val_loader_inf)
            val_preds_list.append(v_preds)
            if val_targets is None: val_targets = v_y
            
            # Predict Test
            t_preds, t_y = trainer.predict(test_loader_inf)
            test_preds_list.append(t_preds)
            if test_targets is None: test_targets = t_y

        # Stack predictions: (N_samples, N_models)
        X_val = np.stack(val_preds_list, axis=1)
        X_test = np.stack(test_preds_list, axis=1)
        
        def get_ensemble_pred(weights, X):
            return np.dot(X, weights)
            
        def objective_fn(weights):
            preds = get_ensemble_pred(weights, X_val)
            if ensemble_metric == "mse": return np.mean((val_targets - preds)**2)
            elif ensemble_metric == "pearson": return -pearsonr(val_targets, preds)[0]
            elif ensemble_metric == "kendall": return -kendalltau(val_targets, preds)[0]
            else: raise ValueError(f"Unknown metric: {ensemble_metric}")

        # Optimization
        n_models = len(seeds)
        # Initialize ensemble weights uniformly (1/N)
        init_weights_opt = np.ones(n_models) / n_models 
        bounds = [(0.0, 1.0)] * n_models
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # SLSQP = Sequential Least Squares Programming
        res = minimize(objective_fn, init_weights_opt, method='SLSQP', bounds=bounds, constraints=constraints)
        best_weights = res.x / np.sum(res.x) if res.success else init_weights
        print(f"Optimized Weights ({ensemble_metric}): {best_weights}")
        
        # Save Weights
        torch.save({"weights": best_weights, "seeds": seeds, "metric": ensemble_metric}, 
                   model_save_dir / f"{timestamp}_{experiment_name}_ensemble_weights.pt")
        
        # Save CSV
        try:
            test_df = pd.read_csv(test_data_csv)
            if len(test_df) != len(test_targets):
                print(f"Warning: Test CSV has {len(test_df)} rows but predictions have {len(test_targets)}. Creating new dataframe.")
                test_df = pd.DataFrame()
        except Exception as e:
            print(f"Could not load test CSV: {e}. Creating new dataframe.")
            test_df = pd.DataFrame()
            
        for idx, seed in enumerate(seeds): test_df[f"pred_seed_{seed}"] = test_preds_list[idx]
        test_df["ensemble_pred"] = get_ensemble_pred(best_weights, X_test)
        if "label" not in test_df.columns: test_df["label"] = test_targets

        test_df.to_csv(predictions_save_dir / f"{timestamp}_{experiment_name}_predictions.csv", index=False)

        # ---------------------------------------------------------
        # Plotting and Stats Saving
        # ---------------------------------------------------------
        
        # Calculate Ensemble Test Metrics
        ensemble_test_preds = get_ensemble_pred(best_weights, X_test)
        ensemble_test_rmse = np.sqrt(np.mean((test_targets - ensemble_test_preds)**2))
        ensemble_test_pearson, _ = pearsonr(test_targets, ensemble_test_preds)
        ensemble_test_kendall, _ = kendalltau(test_targets, ensemble_test_preds)

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
            plt.savefig(plots_dir / f"{experiment_name}_{timestamp}_loss.pdf")
            plt.close()

            # Plot Pearson
            plt.figure(figsize=(10, 6))
            for i, val_pearson in enumerate(all_seeds_val_pearson):
                epochs_range = range(1, len(val_pearson) + 1)
                lbl = 'Validation Pearson' if i == 0 else None
                plt.plot(epochs_range, val_pearson, color='green', alpha=0.3, label=lbl)

            plt.xlabel('Epoch')
            plt.ylabel('Pearson Correlation')
            plt.title(f'Validation Pearson Correlation (All Seeds): {experiment_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / f"{experiment_name}_{timestamp}_pearson.pdf")
            plt.close()

            # Plot Kendall
            plt.figure(figsize=(10, 6))
            for i, val_kendall in enumerate(all_seeds_val_kendall):
                epochs_range = range(1, len(val_kendall) + 1)
                lbl = 'Validation Kendall' if i == 0 else None
                plt.plot(epochs_range, val_kendall, color='purple', alpha=0.3, label=lbl)

            plt.xlabel('Epoch')
            plt.ylabel('Kendall Correlation')
            plt.title(f'Validation Kendall Correlation (All Seeds): {experiment_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / f"{experiment_name}_{timestamp}_kendall.pdf")
            plt.close()

        # Save Per-Epoch Stats for External Plotting
        epoch_stats_rows = []
        for seed_idx, seed in enumerate(seeds):
            # Check if we have data for this seed
            if seed_idx < len(all_seeds_train_loss):
                t_loss = all_seeds_train_loss[seed_idx]
                v_loss = all_seeds_val_loss[seed_idx]
                v_pearson = all_seeds_val_pearson[seed_idx]
                v_kendall = all_seeds_val_kendall[seed_idx]
                
                for ep in range(len(t_loss)):
                    epoch_stats_rows.append({
                        "epoch": ep + 1,
                        "seed": seed,
                        "train_loss": t_loss[ep],
                        "val_loss": v_loss[ep],
                        "val_pearson": v_pearson[ep],
                        "val_kendall": v_kendall[ep]
                    })
        
        pd.DataFrame(epoch_stats_rows).to_csv(save_training_stats_dir / f"{timestamp}_{experiment_name}_epoch_stats.csv", index=False)

        # Save Stats
        stats_file = save_training_stats_dir / "training_stats.csv"
        stats_data = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "test_set_rmse": ensemble_test_rmse,
            "test_set_pearson": ensemble_test_pearson,
            "test_set_kendall": ensemble_test_kendall,
            "training_time_seconds": train_time
        }
        df_stats = pd.DataFrame([stats_data])
        if not stats_file.exists():
            df_stats.to_csv(stats_file, index=False)
        else:
            df_stats.to_csv(stats_file, mode='a', header=False, index=False)

if __name__ == "__main__":
    main()

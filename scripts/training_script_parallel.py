"""
training_script_parallel.py
===========================

Parallel version of training_script.py designed for Slurm array jobs.

It will take a config (list of experiments), an experiment name and a seed and only train the model for that experiment and that seed

Usage:
    python scripts/training_script_parallel.py --config_path config/experiments_config.yml --experiment_name AEV-PLIG --seed 37 --device auto   
"""
import argparse
import torch
import time
from pathlib import Path
import yaml
from tagcn_bind import Trainer, GATv2Net, GATv2Net_v2, TAGCNet, TAGCNet_v2, init_weights, PDBDataset
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

    # Experiment name
    parser.add_argument("--experiment_name", type = str, default="AEV-PLIG", help="The experiment name")

    # Seed
    parser.add_argument("--seed", type = int, default=37, help="Random seed")

    # Run ID
    parser.add_argument("--run_id", type=str, default=None, help="Unique run ID (timestamp) to group files.")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu).")

    return parser.parse_args()

def get_device(device_str):
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str

def get_experiment_config(full_config, experiment_name):

    for exp in full_config["experiments"]:
        if exp["name"] == experiment_name:
            return exp["args"]

    raise ValueError(f"Could not find {experiment_name} in config")

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

    # Get experiment name and corresponding experiment from config
    experiment_name = args.experiment_name

    experiment = get_experiment_config(full_config=config, experiment_name= experiment_name)

    
    

    # Unpack the experiment dict 
    model_name = experiment["model"]
    epochs = experiment["epochs"]
    batch_size = experiment["batch_size"]
    lr = experiment["lr"]
    early_stopping = experiment["early_stopping"]
    early_stopping_metric = experiment["early_stopping_metric"] 
    hidden_dim = experiment["hidden_dim"]
    head = experiment["head"]
    K_param = experiment["K"]
    dropout_rate = experiment["dropout_rate"]
    num_GNN_layers = experiment["num_GNN_layers"]
    activation_function = experiment["activation_function"]
    weight_decay = experiment["weight_decay"] 
    seeds = experiment["seeds"]
    scaling_stats_pt = experiment["scaling_stats_pt"] 
    train_data_csv = experiment["train_data_csv"]
    val_data_csv = experiment["val_data_csv"]
    test_data_csv = experiment["test_data_csv"]  
    train_graphs_dir = experiment["train_graphs_dir"] 
    val_graphs_dir = experiment["val_graphs_dir"]
    test_graphs_dir = experiment["test_graphs_dir"]
    model_save_dir = experiment["model_save_dir"]
    predictions_save_dir = experiment["predictions_save_dir"]  
    save_training_stats_dir = experiment["save_training_stats_dir"]
    plots_dir = experiment["plots_dir"]
    scaling_stats_path = project_root / scaling_stats_pt

    # As we added the standardisation options later we use .get so they default to True for earlier models
    standardise_aevs = experiment.get("standardise_aevs", True)
    standardise_targets = experiment.get("standardise_targets", True)

    # Setup output folders
    model_save_dir = project_root / model_save_dir
    predictions_save_dir = project_root / predictions_save_dir / "rough"
    save_training_stats_dir = project_root / save_training_stats_dir / "rough"
    plots_dir = project_root / plots_dir / "rough"

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
    train_set = PDBDataset(data_dir=train_graphs_dir, dataset_name=train_dataset_name)
    val_set = PDBDataset(data_dir=val_graphs_dir, dataset_name=val_dataset_name)
    test_set = PDBDataset(data_dir=test_graphs_dir, dataset_name=test_dataset_name)

    # Get seed from CLI args
    seed = args.seed

    # Get run id from CLI args for unique file names
    run_id = args.run_id
    
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
    model = initalise_model(model_name=model_name, edge_feat_dim=edge_feat_dim, node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, head=head, num_GNN_layers=num_GNN_layers, activation=activation_function, K=K_param, dropout_rate=dropout_rate)

    # Ensure model is in double precision to match the dataset
    model = model.double()

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

    print(f"Running training for {experiment_name}, model name {model_name}, seed {seed}.")

    # Per seed history
    seed_train_loss = []
    seed_val_loss = []
    seed_train_pearson = []
    seed_val_pearson = []
    seed_train_kendall = []
    seed_val_kendall = []

    for epoch in tqdm(range(epochs), desc="Training Epochs", file=sys.stdout):

        train_loss, train_kendall_corr, train_pearson_corr = trainer.train_epoch(loader=train_loader, optimizer=optimizer, criterion=criterion, standardise_aevs=standardise_aevs, standardise_targets=standardise_targets)
        val_loss, val_kendall_corr, val_pearson_corr = trainer.validate(loader=val_loader, criterion=criterion, standardise_aevs=standardise_aevs, standardise_targets=standardise_targets)
                
        seed_train_loss.append(train_loss)
        seed_val_loss.append(val_loss)
        seed_train_pearson.append(train_pearson_corr)
        seed_val_pearson.append(val_pearson_corr)
        seed_train_kendall.append(train_kendall_corr)
        seed_val_kendall.append(val_kendall_corr)

        tqdm.write(f"Epoch {epoch:03d}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Train Pearson: {train_pearson_corr:.4f} | Val Pearson: {val_pearson_corr:.4f} | Train Kendall: {train_kendall_corr:.4f} | Val Kendall: {val_kendall_corr:.4f}")

        # Check for early stopping, we have a choise to stop on either mse (loss), kendalls tau or pearsons
        if early_stopping_metric == "mse":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save(model.state_dict(), f"{model_save_dir}/{run_id}_{experiment_name}_model_{seed}.pt")
            else:
                wait += 1
                if wait >= early_stopping:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break 
        elif early_stopping_metric == "kendall":
            if val_kendall_corr > best_val_corr:
                best_val_corr = val_kendall_corr
                wait = 0
                torch.save(model.state_dict(), f"{model_save_dir}/{run_id}_{experiment_name}_model_{seed}.pt")
            else:
                wait += 1
                if wait >= early_stopping:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        elif early_stopping_metric == "pearson":
            if val_pearson_corr > best_val_corr:
                best_val_corr = val_pearson_corr
                wait = 0
                torch.save(model.state_dict(), f"{model_save_dir}/{run_id}_{experiment_name}_model_{seed}.pt")
            else:
                wait += 1
                if wait >= early_stopping:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        elif early_stopping_metric == "kendall_rolling":
            low = np.maximum(epoch-7, 0)
            average_kendall = np.mean(seed_val_kendall[low:epoch+1])
            if (average_kendall > best_val_corr):
                best_val_corr = average_kendall
                torch.save(model.state_dict(), f"{model_save_dir}/{run_id}_{experiment_name}_model_{seed}.pt")
            print(f"Current average val kendall is: {average_kendall}, best average val kendall: {best_val_corr}")
 
        elif early_stopping_metric == "pearson_rolling":
            # This is the legacy early stopping method used, as in AEV-PLIG this doesn't actually trigger a traditional break after a certain patience, instead the idea is to
            # run the model for say 200-300 epochs and have it find the best one through this moving average method 
            low = np.maximum(epoch-7, 0)
            average_pearson = np.mean(seed_val_pearson[low:epoch+1])
            if (average_pearson > best_val_corr):
                best_val_corr = average_pearson
                torch.save(model.state_dict(), f"{model_save_dir}/{run_id}_{experiment_name}_model_{seed}.pt")
            print(f"Current average val pearson is: {average_pearson}, best average val pearson: {best_val_corr}")
        
        else:
            raise ValueError(f"Invalid early stopping metric {early_stopping_metric}")
            
    # ---------------------------------------------------------
    # Prediction 
    # ---------------------------------------------------------
    print(f"Training finished for seed {seed}. Starting prediction...")

    train_time = time.time() - start_time      

    # Loaders for inference (no shuffle)
    test_loader_inf = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers = 0, # for test set isn't much point in using multiple workers due to size of data set
        pin_memory = True
    )

        
       
    # Iterate through trained models
    model_path = model_save_dir / f"{run_id}_{experiment_name}_model_{seed}.pt"
            
    # Load model weights into the existing trainer's model
    trainer.model.load_state_dict(torch.load(model_path, weights_only=False))


    # If the model uses dropout it needs to be turned off here:
    if model_name == "GATv2_v2" or model_name == "TAGCN_v2":
        trainer.model.training = False
        
            
    # Predict Test
    test_preds, test_targets = trainer.predict(test_loader_inf, standardise_aevs=standardise_aevs, standardise_targets=standardise_targets)
    
    
    # Calculate Ensemble Test Metrics 
    test_rmse = np.sqrt(np.mean((test_targets - test_preds)**2))
    test_pearson, _ = pearsonr(test_targets, test_preds)
    test_kendall, _ = kendalltau(test_targets, test_preds)

    test_df = pd.DataFrame({
        "test_preds": test_preds,
        "test_targets": test_targets  
    })

    test_df.to_csv(predictions_save_dir / f"{run_id}_{experiment_name}_{seed}_predictions.csv", index = False)

    # ---------------------------------------------------------
    # Plotting and Stats Saving
    # ---------------------------------------------------------
        
    # Plot Loss 
    plt.figure(figsize=(10, 6))
            
    epochs_range = range(1, len(seed_train_loss) + 1)
               
    plt.plot(epochs_range, seed_train_loss, color='blue', alpha=0.3, label="Train loss (MSE)")
    plt.plot(epochs_range, seed_val_loss, color='orange', alpha=0.3, label="Val loss (MSE)")

    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'Training and Validation Loss seed {seed}: {experiment_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{experiment_name}_{run_id}_{seed}_loss.pdf")
    plt.close()

    # Plot Pearson
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(seed_val_pearson) + 1) 
    plt.plot(epochs_range, seed_train_pearson, color="blue", alpha = 0.3, label="Train Pearson")
    plt.plot(epochs_range, seed_val_pearson, color='green', alpha=0.3, label="Val Pearson")

    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.title(f'Validation Pearson Correlation seed {seed}: {experiment_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{experiment_name}_{run_id}_{seed}_pearson.pdf")
    plt.close()

    # Plot Kendall
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(seed_val_kendall) + 1)
    plt.plot(epochs_range, seed_train_kendall, color="blue", alpha=0.3, label="Train kendall")
    plt.plot(epochs_range, seed_val_kendall, color='purple', alpha=0.3, label="Val kendall")

    plt.xlabel('Epoch')
    plt.ylabel('Kendall Correlation')
    plt.title(f'Validation Kendall Correlation seed {seed}: {experiment_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{experiment_name}_{run_id}_{seed}_kendall.pdf")
    plt.close()

    # Save Per-Epoch Stats for External Plotting
    epoch_stats_rows = []        
    for ep in range(len(seed_train_loss)):
        epoch_stats_rows.append({
            "epoch": ep + 1,
            "seed": seed,
            "train_loss": seed_train_loss[ep],
            "val_loss": seed_val_loss[ep],
            "train_pearson": seed_train_pearson[ep],
            "val_pearson": seed_val_pearson[ep],
            "train_kendall": seed_train_kendall[ep],
            "val_kendall": seed_val_kendall[ep]
        })
    # As we want to run in parrallel these csvs must be distinct 
    pd.DataFrame(epoch_stats_rows).to_csv(save_training_stats_dir / f"{run_id}_{experiment_name}_{seed}_epoch_stats.csv", index=False)

    # Save Stats
    stats_file = save_training_stats_dir / f"training_stats_{run_id}_{experiment_name}_{seed}.csv"
    df_stats = pd.DataFrame({
        "timestamp": run_id,
        "experiment_name": experiment_name,
        "seed": seed,
        "test_set_rmse": test_rmse,
        "test_set_pearson": test_pearson,
        "test_set_kendall": test_kendall,
        "training_time_seconds": train_time
    }, index=[0])
    df_stats.to_csv(stats_file)

if __name__ == "__main__":
    main()



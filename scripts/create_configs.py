"""
create_config.py
==================
Generates configs for experiments, processing and plotting with the required parameters
"""

import yaml
import os
import copy

EXPERIMENT_NAME = "experiments_grid_search.yml"
PROCESS_NAME = "process_grid_search.yml"
CASF_PLOT_NAME = "plotting_grid_search_casf.yml"
OOD_PLOT_NAME = "plotting_grid_search_ood.yml"
BUBBLES = True
TRUEVPRED = True
METRICOEPOCH = True # metrics over epochs

TIME_STAMP = "20260306_171000"
TIME_STAMP_OOD = "20260306_171100"

# - name: AEV-PLIG-Intra-Graphs-No-Scale-Rolling
#   args:
#     model: GATv2 
#     epochs: 300 # high epochs for early stopping
#     batch_size: 128
#     lr: 0.001
#     early_stopping: 50
#     early_stopping_metric: "kendall_rolling" # choose mse, kendall, or pearson
#     hidden_dim: 256
#     head: 3
#     K: 3 # Not relavant to GATv2
#     dropout_rate: 0.0 # Not relavant to GATv2
#     num_GNN_layers: 5
#     activation_function: leaky_relu
#     weight_decay: 0.0001 # Explore this
#     seeds: [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]
#     standardise_aevs: False
#     standardise_targets: False
#     scaling_stats_pt: "data/scaling/PDBbind_minus_CASF_2016_scaling.pt" 
#     train_data_csv: "data/PDBbind_processed_train.csv"
#     val_data_csv: "data/PDBbind_processed_val.csv"
#     test_data_csv: "data/CASF_2016_processed.csv"    
#     train_graphs_dir: "data/graphs/PDBbind_train" 
#     val_graphs_dir: "data/graphs/PDBbind_val"
#     test_graphs_dir: "data/graphs/CASF-2016"
#     model_save_dir: "output/models"
#     predictions_save_dir: "output/predictions"
#     save_training_stats_dir: "output/training_stats"
#     plots_dir: "output/plots_dir"

def create_experiment_config(name: str, args: dict):
    """ Creates an experiment config, the args dict will need all of the arguments given above 
    """
    print(args)
    experiment = {
        "name": name,
        "args": {
            "model": args["model"],
            "epochs": args["epochs"],
            "batch_size": args["batch_size"],
            "lr": args["lr"],
            "early_stopping": args["early_stopping"],
            "early_stopping_metric": args["early_stopping_metric"],
            "hidden_dim": args["hidden_dim"],
            "head": args["head"],
            "K": args["K"],
            "dropout_rate": args["dropout_rate"],
            "num_GNN_layers": args["num_GNN_layers"],
            "activation_function": args["activation_function"],
            "weight_decay": args["weight_decay"],
            "seeds": args["seeds"],
            "standardise_aevs": args["standardise_aevs"],
            "standardise_targets": args["standardise_targets"],
            "scaling_stats_pt": args["scaling_stats_pt"],
            "train_data_csv": args["train_data_csv"],
            "val_data_csv": args["val_data_csv"],
            "test_data_csv": args["test_data_csv"],
            "train_graphs_dir": args["train_graphs_dir"],
            "val_graphs_dir": args["val_graphs_dir"],
            "test_graphs_dir": args["test_graphs_dir"],
            "model_save_dir": args["model_save_dir"],
            "predictions_save_dir": args["predictions_save_dir"],
            "save_training_stats_dir": args["save_training_stats_dir"],
            "plots_dir": args["plots_dir"]
        }
    }

    return experiment

def create_process_config(name: str, time_stamp: str, args: dict):
    """ Creates a process config, the args dict will need all of the arguments given above 
    """
    print(args)
    process = {
        "name": name,
        "run_id": time_stamp,
        "args": {
            "model": args["model"],
            "epochs": args["epochs"],
            "batch_size": args["batch_size"],
            "lr": args["lr"],
            "early_stopping": args["early_stopping"],
            "early_stopping_metric": args["early_stopping_metric"],
            "hidden_dim": args["hidden_dim"],
            "head": args["head"],
            "K": args["K"],
            "dropout_rate": args["dropout_rate"],
            "num_GNN_layers": args["num_GNN_layers"],
            "activation_function": args["activation_function"],
            "weight_decay": args["weight_decay"],
            "seeds": args["seeds"],
            "standardise_aevs": args["standardise_aevs"],
            "standardise_targets": args["standardise_targets"],
            "scaling_stats_pt": args["scaling_stats_pt"],
            "train_data_csv": args["train_data_csv"],
            "val_data_csv": args["val_data_csv"],
            "test_data_csv": args["test_data_csv"],
            "train_graphs_dir": args["train_graphs_dir"],
            "val_graphs_dir": args["val_graphs_dir"],
            "test_graphs_dir": args["test_graphs_dir"],
            "model_save_dir": args["model_save_dir"],
            "predictions_save_dir": args["predictions_save_dir"],
            "save_training_stats_dir": args["save_training_stats_dir"],
            "plots_dir": args["plots_dir"]
        }
    }

    return process

def create_plotting_config(name: str, time_stamp: str):
    """ Creates an experiment for the plotting config
    """
    experiment = {
        "name": name,
        "time_stamp": time_stamp

    }
    return experiment

def main():
    # Define the arguments for the configs, some arguments are the same across experiments and even configs
    # for others we will loop through values in our grid search or they will be defined conditionally on the model used

    # Args that remain constant:
    args_base = {
        "epochs": 300,
        "batch_size": 128,
        "lr": 0.001,
        "early_stopping": 100, # this doesn't actually matter if you use the rolling metric
        "early_stopping_metric": "pearson_rolling",
        "activation_function": "leaky_relu",
        "seeds": [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621],
        "standardise_aevs": False,
        "standardise_targets": True,
        "model_save_dir": "output/models",
        "predictions_save_dir": "output/predictions",
        "save_training_stats_dir": "output/training_stats",
        "plots_dir": "output/plots_dir" 
    }

    # We are now going to make arrays of the parameters for the grid search
    # messy grid didn't work due to no weight decay and too low lr:
    # benchmarks = ["CASF-16", "OOD-Test"]
    # graphs = ["intra", "inter"]
    # models = ["TAGCN", "GATv2"]
    # hidden_dim = [256, 512]
    # layers = [1, 3, 5]
    # K_values = [1, 3, 5]
    # heads_values = [1, 3, 5]
    # dropout_values = [0.0, 0.2]
    # Grid search that shows dropout + weight decay essential for OOD Test, time stamp: 20260305_160500 
    # benchmarks = ["CASF-16", "OOD-Test"]
    # graphs = ["intra", "inter"]
    # models = ["TAGCN"]
    # hidden_dim = [512]
    # layers = [5]
    # K_values = [1]
    # heads_values = [1]
    # dropout_values = [0.0, 0.2]
    # weight_decay_values = [0.0, 0.0001]
    # The above will need dropout and weight decay strings added to be compatiable 
    # Grid search to explore higher hidden dim + stronger dropout/weight decay to try and beat SOTA on OOD Test
    benchmarks = ["CASF-16", "OOD-Test"]
    graphs = ["intra", "inter"]
    models = ["TAGCN"]
    hidden_dim = [512, 768]
    layers = [5]
    K_values = [1]
    heads_values = [1]
    dropout_values = [0.2, 0.5]
    dropout_values_strings = ["0-2", "0-5"]
    weight_decay_values = [0.0001, 0.001]
    weight_decay_values_strings = ["0-30-1", "0-20-1"]


    # It is worth pre-defining the arguments for the different graphs and benchmarks
    args_casf_intra = {
        "scaling_stats_pt": "data/scaling/PDBbind_minus_CASF_2016_scaling.pt", 
        "train_data_csv": "data/PDBbind_processed_train.csv",
        "val_data_csv": "data/PDBbind_processed_val.csv",
        "test_data_csv": "data/CASF_2016_processed.csv",    
        "train_graphs_dir": "data/graphs/PDBbind_train", 
        "val_graphs_dir": "data/graphs/PDBbind_val",
        "test_graphs_dir": "data/graphs/CASF-2016" 
    } 
    args_casf_inter = {
        "scaling_stats_pt": "data/scaling/PDBbind_minus_CASF_2016_legacy_scaling.pt", 
        "train_data_csv": "data/PDBbind_processed_train.csv",
        "val_data_csv": "data/PDBbind_processed_val.csv",
        "test_data_csv": "data/CASF_2016_processed.csv",   
        "train_graphs_dir": "data/graphs/PDBbind_legacy_train", 
        "val_graphs_dir": "data/graphs/PDBbind_legacy_val",
        "test_graphs_dir": "data/graphs/CASF-2016_legacy"
    }
    args_ood_test_intra = {
        "scaling_stats_pt": "data/scaling/OOD_Test_scaling.pt", 
        "train_data_csv": "data/OOD_Test_processed_train.csv",
        "val_data_csv": "data/OOD_Test_processed_val.csv",
        "test_data_csv": "data/OOD_Test_processed_test.csv",    
        "train_graphs_dir": "data/graphs/OOD_Test_train", 
        "val_graphs_dir": "data/graphs/OOD_Test_val",
        "test_graphs_dir": "data/graphs/OOD_Test_test"
    }
    args_ood_test_inter = {
        "scaling_stats_pt": "data/scaling/OOD_Test_legacy_scaling.pt", 
        "train_data_csv": "data/OOD_Test_processed_train.csv",
        "val_data_csv": "data/OOD_Test_processed_val.csv",
        "test_data_csv": "data/OOD_Test_processed_test.csv",    
        "train_graphs_dir": "data/graphs/OOD_Test_legacy_train", 
        "val_graphs_dir": "data/graphs/OOD_Test_legacy_val",
        "test_graphs_dir": "data/graphs/OOD_Test_legacy_test"
    }

   
    new_experiments = []
    new_processes = []
    new_plotting_casf = []
    new_plotting_OOD = []
    # mega for loop!
    for benchmark in benchmarks:
        for model in models:
            for graph in graphs:
                for layer in layers:
                    for hidden_dim_val in hidden_dim:
                        for drop_idx, dropout in enumerate(dropout_values):
                            for weight_idx, weight_decay in enumerate(weight_decay_values):
                                for k_head_index in range(0, len(K_values)):
                                    experiment_name = model 
                                    tmp_model = copy.deepcopy(model) 
                                    # If we have a non zero dropout the model name needs to have a _v2 at the end
                                    if dropout > 0.0:
                                        tmp_model += "_v2"
                                        experiment_name += f"-Drop-{dropout_values_strings[drop_idx]}"

                                    experiment_args = {
                                        "model": tmp_model,
                                        "K": K_values[k_head_index],
                                        "head": heads_values[k_head_index]
                                    }


                                    if model[0:3] == "TAG": 
                                        experiment_name += f"-K-{K_values[k_head_index]}"
                                    elif model[0:3] == "GAT":
                                        experiment_name += f"-H-{heads_values[k_head_index]}"                             


                                    
                                    experiment_args["num_GNN_layers"] = layer
                                    experiment_args["hidden_dim"] = hidden_dim_val
                                    experiment_args["dropout_rate"] = dropout
                                    experiment_args["weight_decay"] = weight_decay
                                    
                                    experiment_name += f"-L-{layer}"
                                    experiment_name += f"-Dim-{hidden_dim_val}"
                                    
                                    if weight_decay > 0.0:
                                        experiment_name += f"-WD-{weight_decay_values_strings[weight_idx]}"

                                    if benchmark == "CASF-16" and graph == "intra":
                                        args_bench_args = copy.deepcopy(args_casf_intra)
                                        experiment_name += "-Intra-CASF-2016"
                                        plotting_experiment = create_plotting_config(name=experiment_name, time_stamp=TIME_STAMP)
                                        new_plotting_casf.append(plotting_experiment)
                                    elif benchmark == "CASF-16" and graph == "inter":
                                        args_bench_args = copy.deepcopy(args_casf_inter)
                                        experiment_name += "-Inter-CASF-2016"
                                        plotting_experiment = create_plotting_config(name=experiment_name, time_stamp=TIME_STAMP)
                                        new_plotting_casf.append(plotting_experiment) 
                                    elif benchmark == "OOD-Test" and graph == "intra":
                                        args_bench_args = copy.deepcopy(args_ood_test_intra)
                                        experiment_name += "-Intra-OOD-Test"
                                        plotting_experiment = create_plotting_config(name=experiment_name, time_stamp=TIME_STAMP)
                                        new_plotting_OOD.append(plotting_experiment) 
                                    elif benchmark == "OOD-Test" and graph == "inter":
                                        args_bench_args = copy.deepcopy(args_ood_test_inter)
                                        experiment_name += "-Inter-OOD-Test"
                                        plotting_experiment = create_plotting_config(name=experiment_name, time_stamp=TIME_STAMP)
                                        new_plotting_OOD.append(plotting_experiment)

                                    experiment_args = {**experiment_args, **copy.deepcopy(args_base), **args_bench_args}
                                    
                                    experiment_config = create_experiment_config(name=experiment_name, args=experiment_args) 
                                    new_experiments.append(experiment_config) 
                                    process_config = create_process_config(name=experiment_name, time_stamp=TIME_STAMP, args=experiment_args)
                                    new_processes.append(process_config)
                                

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i, (config_name, configs_to_save) in enumerate(zip([EXPERIMENT_NAME, PROCESS_NAME, CASF_PLOT_NAME, OOD_PLOT_NAME], [new_experiments, new_processes, new_plotting_casf, new_plotting_OOD])):
        # Go up one level to the project root, then into config
        config_path = os.path.join(script_dir, "..", "config", config_name)
        
        # Load existing config if it exists
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {} 
        else:
            # If the config doesn't exist and we are in the plotting case we need to initalise it here
            if i == 2:
                config = {
                    "output_dir": "output/summary_plots",
                    "stats_dir": "output/training_stats",
                    "predictions_dir": "output/predictions",
                    "file_name_tag": TIME_STAMP,
                    "plots": {
                        "bubble": BUBBLES,
                        "metrics_over_epochs": METRICOEPOCH,
                        "true_vs_predicted": TRUEVPRED 
                    }
                }
            elif i == 3:
                # Need a slightly different time stamp for OOD to avoid overlapping file names
                config = {
                    "output_dir": "output/summary_plots",
                    "stats_dir": "output/training_stats",
                    "predictions_dir": "output/predictions",
                    "file_name_tag": TIME_STAMP_OOD,
                    "plots": {
                        "bubble": BUBBLES,
                        "metrics_over_epochs": METRICOEPOCH,
                        "true_vs_predicted": TRUEVPRED 
                    }
                }
            else:
                config = {}
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

        if "experiments" not in config:
            config["experiments"] = []

        # Append new experiments to the existing list
        config["experiments"].extend(configs_to_save)

        # Write back to the file
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)

        print(f"Successfully added {len(configs_to_save)} configs to {config_path}")


if __name__ == "__main__":
    main()

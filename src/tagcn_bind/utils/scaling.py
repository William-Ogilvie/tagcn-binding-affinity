"""
scaling.py
============

This module holds the ScalingManager class which is responsible for scaling the AEVs before training, we will also scale the targets pK. 
We do standard scaling of (x - mean) / std.
""" 

import torch
import numpy as np
import math
from tqdm import tqdm
import os
from typing import List
import sys

class ScalingManager:
    def calculate_stats(self, dataset_paths: List[str], output_path: str, output_file_name: str):
        """ Calculates meand and std for continuous features (the AEVs) and targets (pK), across the whole dataset. 
        That means across all atoms across all ligands in the dataset, so we are getting a global mean and standard deviation.
        As the AEVs are the only continuous feature we just compute it for them. 

        WARNING if you want to add more features to the model you will have to ensure that you keep all the categorical ones up to len_chem_feats
        and all continuous ones after this point in the numpy arrays. Otherwise we will accidently standardise categorical features....

        So that we can just apply this standardisation simply at run time we will do a "fake" standardisation for the categorical features, mean 0 std 1. 

        Args:
            dataset_paths(List[str]): list of paths to the datasets (absolute)
            output_path(str): path to the directory you want to save the stats to (absolute)
            output_file_name(str): name of the output file (excluding .pt)
        """

        print("Starting Global Statistics Calculation")

        # It is important to remember that we save the graphs in shards of 1,000

        # Loop through all the paths in data_set_paths and add to the files array
        shards = []
        i = 0 # Track index of current file path
        for path in dataset_paths:
            temp_files = [(f, i) for f in os.listdir(path) if f.endswith(".pt")]

            shards += temp_files

            i += 1

        n_atoms_total = 0
        # Load the first graph to get the length of the AEV feature vector to initalise sum_aev and sum_aev2
        s_first = shards[0][0]
        file_path = os.path.join(dataset_paths[0], s_first)
        first_shard = torch.load(file_path, weights_only=False)
        exple_idx = list(first_shard.keys())[0]

        uid, graph, pK = first_shard[exple_idx]

        len_chem_feats = graph[4]
        len_aev_feats = graph[5]  

        # Get the dtype of the features array and the pK value
        feat_dtype = graph[1][0].dtype

        # These arrays will track the running total of the AEVs and the AEVs^2 
        sum_aev = np.zeros(len_aev_feats, dtype = feat_dtype)
        sum_aev2 = np.zeros(len_aev_feats, dtype = feat_dtype)

        # Generate the fake standardisation for the categorical features
        fake_mean = np.zeros(len_chem_feats, dtype = feat_dtype)
        fake_std = np.ones(len_chem_feats, dtype = feat_dtype)

        # For the targets we need to know the number of graphs
        num_graphs = 0

        # Initalise empty values for the pK standardisation
        sum_pk = 0.0
        sum_pk2 = 0.0 

        # Loop through the shards
        for shard, i in tqdm(shards, desc = "Processing Shards", file=sys.stdout):
            # Load the shard 
            shard_path = os.path.join(dataset_paths[i], shard)

            shard_file = torch.laod(shard_path, weights_only=False)

            # Loop through the keys of the shard
            for graph_id in tqdm(shard_path.keys(), dec= f"Processing shard {shard_path}", file=sys.stdout):

                # Load graph
                uid, graph, pK = shard_file[graph_id]

                # Update the target sums
                sum_pk += pK
                sum_pk2 += pK ** 2

                # Update the total number of graphs for target average
                num_graphs += 1

                # Get the features vector (technically just a python list) at index 1 of the graph (which is a tuple)
                # Also grab the length of the chemical features (these are our categorical), and the length of the AEVs (these are our numeric/continuous) 
                features = graph[1]
                len_chem_feats = graph[4]
                len_aev_feats = graph[5]


                # Features is a list of numpy arrays, where each array is for each atom in the ligand
                # the first len_chem_feats are chemical features which are categorical, we care about the last 
                # len_aev_feats which are the AEVs
                
                # Loop through features adding the AEVs to the running totals
                for atom in features:

                    # atom will be a numpy array
                    aevs = atom[len_chem_feats:]

                    # Sanity check aevs right length
                    if len(aevs) != len_aev_feats:
                        raise ValueError(f"AEV vector not of expected length for atom of file {file_path}")

                    # add to running total
                    sum_aev += aevs
                    sum_aev2 += aevs**2

                    # update the atom count by 1
                    n_atoms_total += 1

        # Find Global Stats
        mean = sum_aev / n_atoms_total
        variance = (sum_aev2 / n_atoms_total) - (mean**2)
        # The +1e-16 is to ensure we don't divide by 0, so suppose the AEVs are all zero for some column we don't want 
        # a zero standard deviation as will get divide by 0 error. 
        std = np.sqrt(variance + 1e-16)

        # Concatenate these with the fake standardisations
        mean = np.concatenate((fake_mean, mean), axis=0)
        std = np.concatenate((fake_std, std), axis=0)

        # Do the same for the targets
        target_mean = sum_pk / num_graphs
        target_variance = (sum_pk2 / num_graphs) - (target_mean ** 2)
        target_std = math.sqrt(max(0.0, target_variance))

        # Save these stats to the output path
        stats_to_save = {
            'mean': torch.from_numpy(mean).float(),
            'std': torch.from_numpy(std).float(),
            'target_mean': torch.tensor([target_mean]).float(),
            'target_std': torch.tensor([target_std]).float()
        } 
        

        torch.save(stats_to_save, f"{output_path}/{output_file_name}.pt")

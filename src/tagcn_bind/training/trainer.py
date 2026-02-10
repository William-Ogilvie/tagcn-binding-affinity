import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Type
import os
from scipy.stats import pearsonr, kendalltau

class Trainer:

    def __init__(self, model, device, stats_path):
        self.model = model.to(device)
        self.device = device

        # Load the constants from the stats_path
        stats = torch.load(stats_path, weights_only=False)

        self.mean = stats["mean"].to(device)
        self.std = stats["std"].to(device)
        self.target_mean = stats["target_mean"].to(device)
        self.target_std = stats["target_std"].to(device)

    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
 
        for data in loader:
            # data is a "Batch" object from PyTorch Geometric
            data = data.to(self.device)

            # standardise the node features
            data.x = (data.x - self.mean) / self.std

            # Standardise the target
            data.y = (data.y - self.target_mean) / self.target_std

            # reset gradients
            optimizer.zero_grad()

            # forward pass
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            out = out.view(-1)

            # calculate error
            loss = criterion(out, data.y)

            # backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            
            all_preds.append(out.detach().cpu().numpy())
            all_labels.append(data.y.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        kendall_corr, _ = kendalltau(all_labels, all_preds)

        return total_loss / len(loader.dataset), kendall_corr, pearson_corr
    
    @torch.no_grad()
    def validate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for data in loader:
            data = data.to(self.device)

            # standardise
            data.x = (data.x - self.mean) / self.std

            # Standardise the target
            data.y = (data.y - self.target_mean) / self.target_std

            # predict
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out.view(-1), data.y)
            
            total_loss += loss.item() * data.num_graphs
            
            all_preds.append(out.view(-1).cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        pearson_corr, _ = pearsonr(all_labels, all_preds)
        kendall_corr, _ = kendalltau(all_labels, all_preds)

        return total_loss / len(loader.dataset), kendall_corr, pearson_corr

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        for data in loader:
            data = data.to(self.device)
            data.x = (data.x - self.mean) / self.std
            data.y = (data.y - self.target_mean) / self.target_std
            
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            # out now needs to be de standardised
            out = out * self.target_std + self.target_mean
            # Similarly with data.y
            data.y = data.y * self.target_std + self.target_mean
            all_preds.append(out.view(-1).cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
            
        return np.concatenate(all_preds), np.concatenate(all_labels)

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def init_weights(layer):
    """ Xavier Normal initialization for weights, Zeros for bias. """
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)

def calculate_metrics(y_true, y_pred):
    """ Calculates RMSE and Pearson Correlation. """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson, _ = pearsonr(y_true, y_pred)
    
    return rmse, pearson

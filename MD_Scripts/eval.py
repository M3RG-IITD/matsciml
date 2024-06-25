
from __future__ import annotations

import pytorch_lightning as pl
from torch import nn
import torch

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    FrameAveraging,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ScalarRegressionTask
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm

from matsciml.models.utils.io import *
checkpoint_path = "/home/civil/phd/cez218288/scratch/Simulation/trained_models/faenet/model_500k.ckpt"
task = multitask_from_checkpoint(checkpoint_path)

norm_factors = {
    "energy_mean": -186.21986734028704,
    "energy_std": 185.10113806014107,
    "corrected_total_energy_mean": -186.21986734028704,
    "corrected_total_energy_std": 185.10113806014107
}

# Initialize lists for storing predictions and actual values
def initialize_prediction_lists():
    return {
        'Predictions_corr_e': [],
        'Actuals_corr_e': [],
        'Predictions_e': [],
        'Actuals_e': [],
        'Predictions_Fx': [],
        'Actuals_Fx': [],
        'Predictions_Fy': [],
        'Actuals_Fy': [],
        'Predictions_Fz': [],
        'Actuals_Fz': []
    }

# Function to process a data loader
def process_data_loader(data_loader, norm_factors, task, limit=None):
    results = initialize_prediction_lists()

    for batch in tqdm(data_loader):
        Result = task.forward(batch)

        Pred_Energy_corr = Result['regression0']['corrected_total_energy'].item() * norm_factors['energy_std'] + norm_factors['energy_mean']
        Pred_Energy = Result['force_regression0']['energy'].item() * norm_factors['energy_std'] + norm_factors['energy_mean'] 

        Pred_Forces = Result['force_regression0']['force']* norm_factors['energy_std']
        Actual_Energy_corr = batch['targets']['energy'].item()
        Actual_Energy = batch['targets']['corrected_total_energy'].item()
        
        Actual_Forces = batch['targets']['force']

        results['Predictions_corr_e'].append(Pred_Energy_corr)
        results['Actuals_corr_e'].append(Actual_Energy_corr)
        
        results['Predictions_e'].append(Pred_Energy)
        results['Actuals_e'].append(Actual_Energy)
        
        results['Predictions_Fx'] += Pred_Forces[:, 0].reshape(-1).detach().numpy().tolist()
        results['Actuals_Fx'] += Actual_Forces[:, 0].reshape(-1).detach().numpy().tolist()

        results['Predictions_Fy'] += Pred_Forces[:, 1].reshape(-1).detach().numpy().tolist()
        results['Actuals_Fy'] += Actual_Forces[:, 1].reshape(-1).detach().numpy().tolist()

        results['Predictions_Fz'] += Pred_Forces[:, 2].reshape(-1).detach().numpy().tolist()
        results['Actuals_Fz'] += Actual_Forces[:, 2].reshape(-1).detach().numpy().tolist()

        if limit and len(results['Predictions_corr_e']) >= limit:
            break

    return results

# Load data
dm = MatSciMLDataModule(
    "MaterialsProjectDataset",
    test_split="/home/civil/phd/cez218288/scratch/Scaling_lmdb_new/test_new",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(cutoff_radius=6.0, adaptive_cutoff=True),
            PointCloudToGraphTransform(
                "pyg",
                node_keys=["pos", "atomic_numbers"],
            ),
            FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
        ],
    },
    batch_size=1,
)

dm.setup()
test_loader = dm.test_dataloader()

# Process test data
test_results = process_data_loader(test_loader, norm_factors, task, limit=100000)

# Function to plot R² score
def plot_r2_score(actual, pred, title="Title", suffix="500k_Faenet"):
    save_dir = '/home/civil/phd/cez218288/scratch/Simulation/plots/'

    # Calculate R² score
    r2 = r2_score(actual, pred)

    # Create the scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, pred, label='Test', color='green')
    plt.xlabel("Actual", fontsize=20, fontweight='bold')
    plt.ylabel("Predicted", fontsize=20, fontweight='bold')
    plt.xticks(rotation=90, fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    min_val = min(min(actual), min(pred))
    max_val = max(max(actual), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.title(title, fontsize=20, fontweight='bold')

    # Annotate the R² score on the plot
    plt.text(0.05, 0.95, f'R² = {r2:.5f}', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', verticalalignment='top', color='green')

    # Save the plot
    filename = f"{title.replace(' ', '_')}_{suffix}.png"
    plt.savefig(f'{save_dir}{filename}', bbox_inches='tight')
    plt.show()
    plt.clf()

# Plot and save the results for test data
plot_r2_score(test_results['Actuals_corr_e'], test_results['Predictions_corr_e'], "Corrected Total Energy", "500k_Faenet_trail")
plot_r2_score(test_results['Actuals_e'], test_results['Predictions_e'], "Energy", "500k_Faenet_trail")
plot_r2_score(test_results['Actuals_Fx'], test_results['Predictions_Fx'], "Fx", "500k_Faenet_trail")
plot_r2_score(test_results['Actuals_Fy'], test_results['Predictions_Fy'], "Fy", "500k_Faenet_trail")
plot_r2_score(test_results['Actuals_Fz'], test_results['Predictions_Fz'], "Fz", "500k_Faenet_trail")

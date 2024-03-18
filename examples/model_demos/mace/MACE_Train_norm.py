# %%
from __future__ import annotations

import argparse
import sys

# sys.path.append("/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/")

import e3nn
# import math
# Atomic Energies table
import mendeleev
import pytest
import pytorch_lightning as pl
from mendeleev.fetch import fetch_ionization_energies
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.loggers import WandbLogger
# from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm

from matsciml.datasets import transforms

# sys.path.append(
#     "/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/",
# )  # Path to matsciml directory(or matsciml installed as package )
from matsciml.datasets.lips import LiPSDataset, lips_devset
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning.callbacks import GradientCheckCallback
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import MaceEnergyForceTask
from matsciml.models.pyg.mace import data, modules, tools
from matsciml.models.pyg.mace.modules.blocks import *
from matsciml.models.pyg.mace.modules.models import ScaleShiftMACE
from matsciml.models.pyg.mace.modules.utils import compute_mean_std_atomic_inter_energy
from matsciml.models.pyg.mace.tools import atomic_numbers_to_indices, to_one_hot

pl.seed_everything(6)

torch.manual_seed(111)
# %%


# atomic_energies = fetch_ionization_energies(degree=list(range(1, 5))).sum(axis=1)
# atomic_energies *= -1
# atomic_energies = torch.Tensor(list(atomic_energies[:119].to_dict().values()))


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


### Gnome
# pre_compute_params = {
#     "mean": 64690.4765625,
#     "std": 42016.30859375,
#     "avg_num_neighbors": 25.7051,
# }
# DATASET = "gnome"
# TRAIN_PATH = "/store/code/open-catalyst/data_lmdbs/gnome/train"
# VAL_PATH = "/store/code/open-catalyst/data_lmdbs/gnome/val"


### MP-Traj
# pre_compute_params = {
#     "mean": 27179.298828125,
#     "std": 28645.603515625,
#     "avg_num_neighbors": 52.0138,
# }
# DATASET = "mp-traj"
# TRAIN_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj/train"
# VAL_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj/val"


import torch
import torch.nn as nn
import torch.optim as optim



DATASET = "gnome_mptraj"
# TRAIN_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj-gnome-combo/train"
# VAL_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj-gnome-combo/val"
TRAIN_PATH =  "/home/m3rg2000/matsciml/matsciml/datasets/devset" #"/home/m3rg2000/matsciml/matsciml/datasets/devset" #"matsciml/datasets/lips/devset"#"/datasets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/train"
VAL_PATH = "/home/m3rg2000/matsciml/matsciml/datasets/trial/test" #"matsciml/datasets/lips/devset"#"/datasesets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/val"

def compute_average_E0s(collections_train, z_table,max_iter=50):
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s

    E = C1*(N_AtmicN1)+C2*(N_Atm2) ... Ck*(N_Atmk)
    E : Frame Actual Energy
    C1,C2,...,Cn : Average E0

    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    # Convert arrays to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability
    A = torch.zeros(max_iter, len_zs)
    B = torch.zeros(max_iter)
    
    for i, batch in enumerate(collections_train):
        if(i>max_iter-1):
            break
        B[i] = batch['energy'].sum()  
        for j, z in enumerate(z_table):
            A[i, j] = torch.count_nonzero(batch['graph']['atomic_numbers'][:] == z)  
        
    A = A.to(device)  
    B = B.to(device)  

    # Solve the linear system using PyTorch
    E0s = torch.linalg.lstsq(A, B)[0]
    
    return E0s.cpu().numpy()  # Move result back to CPU and convert to NumPy array

 
def process_batch(batch, atomic_energies_fn):
    graph = batch.get("graph")
    atomic_numbers = getattr(graph, "atomic_numbers")
    z_table = torch.arange(1, 118 + 1)
    indices = atomic_numbers - 1
    node_attrs = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table)
    )
    node_e0 = atomic_energies_fn(node_attrs)
    graph_e0s = scatter_sum(
        src=node_e0, index=graph.batch, dim=-1, dim_size=graph.num_graphs
    )
    graph_sizes = graph.ptr[1:] - graph.ptr[:-1]

    atomic_inter_energy = (batch['energy'] - graph_e0s) / graph_sizes
    mean = atomic_inter_energy.mean().item()
    std = atomic_inter_energy.std().item()

    avg_num_neighbors = graph.edge_index.numel() / len(atomic_numbers)

    return mean, std, avg_num_neighbors


def compute_mean_std_atomic_inter_energy_and_avg_num_neighbors(
    data_loader: torch.utils.data.DataLoader, atomic_energies, convergence_threshold=0.1, max_iterations=50
):
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)
    
    mean_estimate = 0
    std_estimate = 0
    iterations = 0
    means=[]
    for batch in data_loader:
        mean, std, avg_num_neighbors = process_batch(batch, atomic_energies_fn)
        
        # Update running estimates
        iterations += 1
        delta_mean = mean - mean_estimate
        delta_std = std - std_estimate
        mean_estimate += delta_mean / iterations
        std_estimate += delta_std / iterations
        print(iterations,mean,std,delta_mean,delta_std)
        means+=[mean_estimate]
        if iterations > 1 and abs(delta_mean) < convergence_threshold and abs(delta_std) < convergence_threshold:
            print("Converged")
            return mean_estimate, std_estimate, avg_num_neighbors
        
        if iterations >= max_iterations:
            print("Max Iterations reached")
            return mean_estimate, std_estimate, avg_num_neighbors
    print("Data Exhausted")
    return mean_estimate, std_estimate, avg_num_neighbors



# %%
def main(args):
    torch.autograd.set_detect_anomaly(True)

    # Load Data
    dm = MatSciMLDataModule(
        "MaterialsProjectDataset",
        train_path=TRAIN_PATH,
        val_split=VAL_PATH,
        test_split=VAL_PATH,
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(cutoff_radius=10.0),
                PointCloudToGraphTransform("pyg", cutoff_dist=10.0),#args.r_max),
            ],
        },
        batch_size=16,
    )

    dm.setup()
    train_loader = dm.train_dataloader()
    dataset_iter = iter(train_loader)
    batch = next(dataset_iter)
    
    atomic_energies=torch.ones(118)*(-5)
    atomic_energies=compute_average_E0s(train_loader,np.arange(1,119))
    atomic_inter_shift,atomic_inter_scale,avg_num_neighbors =compute_mean_std_atomic_inter_energy_and_avg_num_neighbors(train_loader,atomic_energies)
    # atomic_inter_shift,atomic_inter_scale,avg_num_neighbors = -10, 1, 15

    atomic_numbers = torch.arange(1, 119)

    # atomic_inter_shift = pre_compute_params["mean"]
    # atomic_inter_scale = pre_compute_params["std"]
    # avg_num_neighbors = pre_compute_params["avg_num_neighbors"]
    
    print("atomic_inter_shift",atomic_inter_shift)
    print("atomic_inter_scale",atomic_inter_scale)
    print("avg_num_neighbors",avg_num_neighbors)

    # Load Model
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_bessel,
        num_polynomial_cutoff=args.num_polynomial_cutoff,
        max_ell=args.max_ell,
        # max_L=args.max_L,
        # num_channels=args.num_channels,
        # num_radial_basis=args.num_radial_basis,
        # scaling=args.scaling,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        num_interactions=args.num_interactions,
        num_elements=len(atomic_numbers),
        hidden_irreps=e3nn.o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=args.correlation_order,
        gate=torch.nn.functional.silu,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        MLP_irreps=e3nn.o3.Irreps(args.MLP_irreps),
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
        training=True,
    )
    def custom_lr_schedule(warmup_epochs=10, decay_epochs=30, total_epochs=100):
        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                # Increase the learning rate linearly during the warmup phase
                return 0.1+current_epoch / warmup_epochs
            elif current_epoch < warmup_epochs + decay_epochs:
                # Keep the learning rate constant
                return 1.0
            else:
                # Decrease the learning rate linearly after warmup + decay epochs
                return 1 - (current_epoch - warmup_epochs - decay_epochs) / (total_epochs - warmup_epochs - decay_epochs)

        
        # scheduler = LambdaLR(optimizer, lr_lambda)
        return lr_lambda
    task = MaceEnergyForceTask(
        encoder_class=ScaleShiftMACE,
        encoder_kwargs=model_config,
        task_keys=["energy", "force"],
        # scheduler_kwargs= {'CosineAnnealingLR':{"T_max":100,"optimizer":optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4},},
        scheduler_kwargs = {
            'LambdaLR': {
            'lr_lambda': custom_lr_schedule,
            
            }
        },
        output_kwargs={
            "energy": {
                "block_type": "IdentityOutputBlock",
                "output_dim": 1,
                "hidden_dim": None,
            },
            "force": {
                "block_type": "IdentityOutputBlock",
                "output_dim": 3,
                "hidden_dim": None,
            },
        },
        loss_coeff={"energy": 1.0, "force": 10.0},
        lr=0.00001,
        weight_decay=1e-8,
    )

    # Print model
    print(task)

    # Start Training
    # logger = CSVLogger(save_dir="./mace_experiments")
    wandb.init(project='normalisation', entity='m3rg', mode='online', name="custom_normal_re1")
    logger = WandbLogger(log_model="all", name=f"mace-{DATASET}-data", save_dir='./Trial_Mace')

    mc = ModelCheckpoint(monitor="val_energy", save_top_k=5)
    # accumulator = GradientAccumulationScheduler(scheduling={0: 1})
    
    trainer = pl.Trainer(
        max_epochs=1000,
        min_epochs=20,
        log_every_n_steps=5,
        precision=16,
        accelerator="gpu",
        # limit_train_batches=0.8, 
        # limit_val_batches=0.1, 
        
        devices=1,
        gradient_clip_val=0.1,
        # strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[
            GradientCheckCallback(),
            mc,
            # accumulator,
        ],
    )

    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACE Training script")
    parser.add_argument("--r_max", type=float, default=6.0, help="Neighbor cutoff")
    parser.add_argument("--max_ell", type=int, default=3, help="Spherical harmonic Lmax")
    parser.add_argument(
        "--num_bessel",
        type=int,
        default=3,
        help="Bessel embeding size",
    )
    parser.add_argument(
        "--num_polynomial_cutoff",
        type=int,
        default=5,
        help="Radial basis polynomial cutoff",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=2,
        help="No. of interaction layers",
    )
    parser.add_argument(
        "--hidden_irreps",
        type=str,
        default="16x0e+16x1o+16x2e",
        help="Hidden Irrep Shape",
    )
    parser.add_argument(
        "--correlation_order",
        type=int,
        default=3,
        help="Correlation Order",
    )
    parser.add_argument(
        "--MLP_irreps",
        type=str,
        default="16x0e",
        help="Irreps of Non-linear readout block",
    )

    parser.add_argument(
        "--max_L",
        type=int,
        default=2,
        help="max_L",
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=128,
        help="num_channels",
    )

    parser.add_argument(
        "--num_radial_basis",
        type=int,
        default=10,
        help="num_radial_basis",
    )

    parser.add_argument(
        "--scaling",
        type=str,
        default="rms_forces_scaling",
        help="scaling parameter",
    )

    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")

    args = parser.parse_args()
    main(args)



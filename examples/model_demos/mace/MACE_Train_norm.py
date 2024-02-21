# %%
from __future__ import annotations

import argparse
import sys

# sys.path.append("/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/")

import e3nn

# Atomic Energies table
import mendeleev
import pytest
import pytorch_lightning as pl
from mendeleev.fetch import fetch_ionization_energies
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
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

def compute_average_E0s(
    collections_train, z_table
):
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s

    E = C1*(N_AtmicN1)+C2*(N_Atm2) ... Ck*(N_Atmk)
    E : Frame Actual Energy
    C1,C2,...,Cn : Average E0

    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    print(A.shape)
    i=0
    for batch in collections_train:
        B[i] = batch['energy'].sum()
        
        for j, z in enumerate(z_table):
            A[i, j] = np.count_nonzero(batch['graph']['atomic_numbers'][:] == z)
        i+=1
    E0s = np.linalg.lstsq(A, B, rcond=None)[0]
    return E0s

 

### Combined Datasets
# pre_compute_params = {
#     "mean": 150,
#     "std": 50,
#     "avg_num_neighbors": 20,
# }

DATASET = "gnome_mptraj"
# TRAIN_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj-gnome-combo/train"
# VAL_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj-gnome-combo/val"
TRAIN_PATH = "/home/m3rg2000/matsciml/matsciml/datasets/trial/devset" #"matsciml/datasets/lips/devset"#"/datasets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/train"
VAL_PATH = "/home/m3rg2000/matsciml/matsciml/datasets/trial/test" #"matsciml/datasets/lips/devset"#"/datasesets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/val"
def compute_mean_std_atomic_inter_energy_and_avg_num_neighbors(
    data_loader: torch.utils.data.DataLoader,atomic_energies,
):
    #avg_atom_inter_es_list = []
    avg_num_neighbors_list=[]
    mean_sum=0
    std_sum=0
    counter=0
    
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    for batch in data_loader:
        counter+=1
       
        atomic_energies =atomic_energies
        graph = batch.get("graph")
        atomic_numbers: torch.Tensor = getattr(graph, "atomic_numbers")
        z_table=torch.arange(1,len(atomic_energies)+1)#List of atomic numbers [1,...,118] #tools.get_atomic_number_table_from_zs(atomic_numbers.numpy())
        indices = atomic_numbers-1 # Index of atomic number in z_table #atomic_numbers_to_indices(atomic_numbers, z_table=z_table)
        node_attrs = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table))
        node_e0 = atomic_energies_fn(node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=graph.batch, dim=-1, dim_size=graph.num_graphs
        )
        graph_sizes = graph.ptr[1:] - graph.ptr[:-1]
        # avg_atom_inter_es_list.append(
        #     (batch['energy'] - graph_e0s) / graph_sizes
        # )  # {[n_graphs], }
        avg_num_neighbors_list.append(graph.edge_index.numel()/len(atomic_numbers))
       
        mean = to_numpy(torch.mean((batch['energy'] - graph_e0s) / graph_sizes)).item()
        std = to_numpy(torch.std((batch['energy'] - graph_e0s) / graph_sizes)).item()
        mean_sum+=mean
        std_sum+=std
        print(mean, std)
    avg_num_neighbors= torch.mean(torch.Tensor(avg_num_neighbors_list))
    return mean_sum/counter, std_sum/counter, avg_num_neighbors

# %%
def main(args):
    
    # Load Data
    dm = MatSciMLDataModule(
        "MaterialsProjectDataset",
        train_path=TRAIN_PATH,
        val_split=VAL_PATH,
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
    
    
    atomic_energies=compute_average_E0s(train_loader,np.arange(1,119))
    atomic_inter_shift,atomic_inter_scale,avg_num_neighbors =compute_mean_std_atomic_inter_energy_and_avg_num_neighbors(train_loader,atomic_energies)
   
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

    task = MaceEnergyForceTask(
        encoder_class=ScaleShiftMACE,
        encoder_kwargs=model_config,
        task_keys=["energy", "force"],
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
        lr=0.005,
        weight_decay=1e-8,
    )

    # Print model
    print(task)

    # Start Training
    # logger = CSVLogger(save_dir="./mace_experiments")
    wandb.init(project='normalisation', entity='m3rg', mode='online', )
    logger = WandbLogger(log_model="all", name=f"mace-{DATASET}-data", save_dir='./Trial_Mace')

    mc = ModelCheckpoint(monitor="val_force", save_top_k=5)

    trainer = pl.Trainer(
        max_epochs=1000,
        min_epochs=20,
        log_every_n_steps=5,
        accelerator="gpu",
        devices=1,
        # strategy="ddp_find_unused_parameters_true",
        logger=logger,
        # callbacks=[
        #     GradientCheckCallback(),
        #     mc,
        # ],
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



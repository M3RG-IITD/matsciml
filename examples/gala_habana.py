# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import pytorch_lightning as pl

from ocpmodels.datasets import s2ef_devset, is2re_devset, S2EFDataset
from ocpmodels.models import S2EFPointCloudModule, GalaPotential
from ocpmodels.lightning.data_utils import PointCloudDataModule
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

BATCH_SIZE = 1
NUM_WORKERS = 0
REGRESS_FORCES = True

MAX_STEPS = 5


# default model configuration for DimeNet++
model_config = {
          "D_in": 200,
          "hidden_dim": 128,
          "merge_fun": "concat",
          "join_fun": "concat",
          "invariant_mode": "full",
          "covariant_mode": "full",
          "include_normalized_products": True,
          "invar_value_normalization": "momentum",
          "eqvar_value_normalization": "momentum_layer",
          "value_normalization": "layer",
          "score_normalization": "layer",
          "block_normalization": "layer",
          "equivariant_attention": False,
          "tied_attention": True
        }

# use default settings for DimeNet++
gnn = GalaPotential(**model_config)

print('S2EF Training')

# use the GNN in the LitModule for all the logging, loss computation, etc.
model = S2EFPointCloudModule(gnn, regress_forces=REGRESS_FORCES, lr=1e-3, gamma=0.1)

data_module = PointCloudDataModule(
    train_path="./ocpmodels/datasets/dev-min",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    dataset_class=S2EFDataset
)


# default is TensorBoardLogger, but here we log to CSV for illustrative
# purposes; see link below for list of supported loggers:
# https://pytorch-lightning.readthedocs.io/en/1.6.3/extensions/logging.html
logger = CSVLogger("./lightning_logs")

# callbacks are passed as a list into `Trainer`; see link below for API
# https://pytorch-lightning.readthedocs.io/en/1.6.3/extensions/callbacks.html
ckpt_callback = ModelCheckpoint("model_checkpoints", save_top_k=5, monitor="train_total")

trainer = pl.Trainer(
    accelerator="cpu",
    logger=logger,
    callbacks=[ckpt_callback, ModelSummary(max_depth=2)],
    max_steps=MAX_STEPS,
    log_every_n_steps=1)

trainer.fit(model, datamodule=data_module)
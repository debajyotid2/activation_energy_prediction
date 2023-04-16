"""
Reaction activation energy prediction using PyTorch.
"""

import time
import logging

from typing import Any, Optional
from pathlib import Path

import pandas as pd
import torch
import pytorch_lightning as pl

from grambow_dataset import pytorch_dataset
from model import MLPRegressor


NUM_LAYERS = 2
NUM_NODES_PER_LAYER = 200
LEARNING_RATE = 0.001

DATA_DIRPATH = "../data"

RADIUS = 5
N_BITS = 1024

BATCH_SIZE = 32

# Adjust depending on the number of CPU cores in your machine.

NUM_WORKERS = 0

# Percentage of the whole dataset utilized
DATASET_FRAC = 1.0

# Data is split as : Test = Test frac; (Train+Val) = (1-Test frac)
#                  : Val = Val frac * (1-Test frac)
#                  : Train = (1-Val frac) * (1-Test frac)
# Before changing VAL_FRAC and TEST_FRAC, please delete all
# npz files from data/compressed.

VAL_FRAC = 0.1      
TEST_FRAC = 0.2

MAX_EPOCHS = 5
SEED = 42

logging.basicConfig(
            format="%(asctime)s-%(levelname)s: %(message)s",
            level=logging.DEBUG
        )

def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")

    data_dir = Path(DATA_DIRPATH)

    dataset_frac = DATASET_FRAC

    start_time = time.perf_counter()
    logging.info("Loading data...")
    train_loader, val_loader, test_loader = \
            pytorch_dataset.load_dataloaders_1(
                            data_dirpath=data_dir,
                            radius=RADIUS,
                            n_bits=N_BITS,
                            batch_size=BATCH_SIZE,
                            val_frac=VAL_FRAC,
                            test_frac=TEST_FRAC,
                            num_workers=NUM_WORKERS,
                            dataset_frac=dataset_frac,
                            seed=SEED
                        )
    logging.info(f"Loaded data in {time.perf_counter()-start_time:.3f} seconds.")

    start_time = time.perf_counter()
    model = MLPRegressor(
                num_input_nodes=N_BITS+1,
                num_layers=NUM_LAYERS,
                num_nodes_per_layer=NUM_NODES_PER_LAYER,
                learning_rate=LEARNING_RATE,
                loss_function=torch.nn.MSELoss()
            )
    logging.info(f"Initialized model in {time.perf_counter()-start_time:.3f} seconds.")
    
    start_time = time.perf_counter()
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         logger=True, 
                         enable_progress_bar=True,
                         max_epochs=MAX_EPOCHS)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    logging.info(f"Finished training model in {time.perf_counter()-start_time:.3f} seconds.")
    
    # model.eval()

    # l1_loss = torch.nn.L1Loss(reduction="sum")
    # l2_loss = torch.nn.MSELoss(reduction="sum")
    # 
    # mae_loss_train, mae_loss_test = 0.0, 0.0
    # mse_loss_train, mse_loss_test = 0.0, 0.0

    # y_train, y_test = torch.zeros((1)), torch.zeros((1))
    # for batch in iter(train_loader):
    #     x, y = batch
    #     y_pred = model(x)
    #     y_train = torch.cat((y_train, y), 0)
    #     mae_loss_train += l1_loss(y_pred.view(-1), y)
    #     mse_loss_train += l2_loss(y_pred.view(-1), y)

    # for batch in iter(test_loader):
    #     x, y = batch
    #     y_pred = model(x)
    #     y_test = torch.cat((y_test, y), 0)
    #     mae_loss_test += l1_loss(y_pred.view(-1), y)
    #     mse_loss_test += l2_loss(y_pred.view(-1), y)
    # 
    # y_train = y_train[1:]
    # y_test = y_test[1:]
    # 

    # r_sq_train = 1-mse_loss_train/(l2_loss(y_train, torch.ones_like(y_train)*torch.mean(y_train, 0)))
    # r_sq_test = 1-mse_loss_test/(l2_loss(y_test, torch.ones_like(y_test)*torch.mean(y_test, 0)))
    # mse_loss_train /= len(y_train)
    # mse_loss_test /= len(y_test)
    # mae_loss_train /= len(y_train)
    # mae_loss_test /= len(y_test)

if __name__ == "__main__":
    main()

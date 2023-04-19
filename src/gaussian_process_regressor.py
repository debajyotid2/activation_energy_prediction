"""
Activation energy predictor model on the Grambow dataset using
scikit-learn.
"""

from pathlib import Path
from typing import Any, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import gpytorch
import torchmetrics

from grambow_dataset import grambow, pytorch_dataset

logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

SEED = 42

BATCH_SIZE = 32
TEST_FRAC = 0.2

RADIUS = 5
N_BITS = 1024

MAX_EPOCHS = 1
LEARNING_RATE = 0.001

DATA_PATH = "../data/b97d3.csv"


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood: Any):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_x, covar_x = self.mean_module(x), self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main() -> None:
    data_path = Path(DATA_PATH)
    max_epochs = MAX_EPOCHS

    # load data
    X_train, Y_train, X_test, Y_test = \
            grambow.load_data_random_split_2(data_path=data_path,
                                radius=RADIUS,
                                test_frac=TEST_FRAC,
                                n_bits=N_BITS)

    train_ds = pytorch_dataset.GrambowDataset(X_train, Y_train) 
    test_ds = pytorch_dataset.GrambowDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    
    # model definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = ExactGPModel(torch.from_numpy(train_ds.X), 
                         torch.from_numpy(train_ds.Y), 
                         likelihood)

    likelihood.to(device)
    model.to(device)

    loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training
    logging.info("Starting training...")
     
    model.train()
    likelihood.train()
    
    for epoch in range(max_epochs):
        for X_train, Y_train in train_loader:   
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)
            
            optimizer.zero_grad()
            breakpoint()
            Y_pred = model(X_train)
            loss = -loss_fn(Y_pred, Y_train)
            loss.backward()
            logging.debug(f"Epoch: {epoch}, loss: {loss.item():.3f}, "+
              f"lengthscale: "+
              f"{model.covar_module.base_kernel.lengthscale.item():.3f}, "+
              f"noise: {model.likelihood.noise.item():.3f}")
            optimizer.step()
   
    logging.info(f"Finished training.")

    # test
    model.eval()
    likelihood.eval()
    
    with torch.nograd(), gpytorch.settings.fast_pred_var():
        Y_pred = likelihood(model(X_test))
    logging.info(f"Test R-squared = "+\
            f"{torchmetrics.R2Score()(Y_pred, Y_test):.4f}")

    mae_test = torchmetrics.MeanAbsoluteError()(Y_pred, Y_test)

    logging.info(f"Test MAE = {mae_test:.4f} kcal/mol")

if __name__ == "__main__":
    main()


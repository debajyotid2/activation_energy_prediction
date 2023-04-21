"""
Activation energy predictor model on the Grambow dataset using
scikit-learn.
"""

from pathlib import Path
from typing import Any, Callable, Optional
import logging
import sys

sys.path.insert(0, Path.cwd().resolve().parents[1])
breakpoint()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.linear_model

from src.rgd1_dataset import rgd1


logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

SEED = 42

VAL_FRAC = 0.1
TEST_FRAC = 0.2

RADIUS = 5
N_BITS = 1024

DATA_PATH = "../../data"

def run_regression(model: Callable[Any, Any],
                   X_train: np.ndarray[Any, Any],
                   Y_train: np.ndarray[Any, Any],
                   X_test: np.ndarray[Any, Any],
                   Y_test: np.ndarray[Any, Any]) -> None:
    # training
    logging.info("Starting training...")
    model.fit(X_train, Y_train)

    logging.info(f"Finished training.")
    logging.info(f"Test R-squared = {model.score(X_test, Y_test):.4f}")

    Y_pred = model.predict(X_test)
    mae_test = np.mean(np.abs(Y_pred-Y_test), axis=0)
    mse_test = np.mean((Y_pred-Y_test)**2, axis=0)

    logging.info(f"Test MAE = {mae_test:.4f} kcal/mol")
    logging.info(f"Test MSE = {mse_test:.4f} kcal/mol")

def main() -> None:
    data_path = Path(DATA_PATH)

    # load data
    X_train, Y_train, X_test, Y_test = \
            rgd1.load_data_random_split(
                                data_path=data_path,
                                radius=RADIUS,
                                test_frac=TEST_FRAC,
                                n_bits=N_BITS,
                                seed=SEED)
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = \
    #         grambow.load_data_scaffold_split(
    #                             data_path=data_path,
    #                             radius=RADIUS,
    #                             val_frac=VAL_FRAC,
    #                             test_frac=TEST_FRAC,
    #                             n_bits=N_BITS)
    # X_train = np.vstack([X_train, X_val])
    # Y_train = np.hstack([Y_train, Y_val])

    # linear regression
    logging.info("LINEAR REGRESSION")
    model = sklearn.linear_model.LinearRegression()
    run_regression(model, X_train, Y_train, X_test, Y_test)

    # ridge regression
    logging.info("RIDGE REGRESSION")
    model = sklearn.linear_model.Ridge(random_state=SEED)
    run_regression(model, X_train, Y_train, X_test, Y_test)

    # lasso regression
    logging.info("LASSO REGRESSION")
    model = sklearn.linear_model.Lasso(random_state=SEED)
    run_regression(model, X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
    main()


"""
Activation energy predictor model on the Grambow dataset using
scikit-learn.
"""

from pathlib import Path
from typing import Any, Callable, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.linear_model

from grambow_dataset import grambow

logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

SEED = 42

TEST_FRAC = 0.2

RADIUS = 5
N_BITS = 1024

DATA_PATH = "../data/b97d3.csv"

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

    logging.info(f"Test MAE = {mae_test:.4f} kcal/mol")

def main() -> None:
    data_path = Path(DATA_PATH)

    # load data
    X_train, Y_train, X_test, Y_test = \
            grambow.load_data_2(data_path=data_path,
                                radius=RADIUS,
                                test_frac=TEST_FRAC,
                                n_bits=N_BITS)

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


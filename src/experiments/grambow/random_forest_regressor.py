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

import sklearn.ensemble

from grambow_dataset import grambow

logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

SEED = 42

VAL_FRAC = 0.1
TEST_FRAC = 0.2

RADIUS = 5
N_BITS = 1024

NUM_ESTIMATORS = 200

DATA_PATH = "../data/b97d3.csv"

def main() -> None:
    data_path = Path(DATA_PATH)

    # load data
    # X_train, Y_train, X_test, Y_test = \
    #         grambow.load_data_random_split_2(data_path=data_path,
    #                             radius=RADIUS,
    #                             test_frac=TEST_FRAC,
    #                             n_bits=N_BITS)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
            grambow.load_data_scaffold_split(
                                data_path=data_path,
                                radius=RADIUS,
                                val_frac=VAL_FRAC,
                                test_frac=TEST_FRAC,
                                n_bits=N_BITS)
    X_train = np.vstack([X_train, X_val])
    Y_train = np.hstack([Y_train, Y_val])

    # training
    logging.info("Starting training...")
    model = sklearn.ensemble.RandomForestRegressor(
                                    n_estimators=NUM_ESTIMATORS,
                                    n_jobs=4,
                                    random_state=SEED)
    model.fit(X_train, Y_train)

    logging.info(f"Finished training.")
    logging.info(f"Test R-squared = {model.score(X_test, Y_test):.4f}")

    Y_pred = model.predict(X_test)
    mae_test = np.mean(np.abs(Y_pred-Y_test), axis=0)
    mse_test = np.mean((Y_pred-Y_test)**2, axis=0)

    logging.info(f"Test MAE = {mae_test:.4f} kcal/mol")
    logging.info(f"Test MSE = {mse_test:.4f} kcal/mol")

if __name__ == "__main__":
    main()


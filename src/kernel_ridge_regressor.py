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

import sklearn.kernel_ridge

from grambow_dataset import grambow

logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

SEED = 42

TEST_FRAC = 0.2

RADIUS = 5
N_BITS = 1024

DATA_PATH = "../data/b97d3.csv"

def main() -> None:
    data_path = Path(DATA_PATH)

    # load data
    X_train, Y_train, X_test, Y_test = \
            grambow.load_data_random_split_1(data_path=data_path,
                                radius=RADIUS,
                                test_frac=TEST_FRAC,
                                n_bits=N_BITS)

    # training
    logging.info("Starting training...")
    model = sklearn.kernel_ridge.KernelRidge()
    model.fit(X_train, Y_train)

    logging.info(f"Finished training.")
    logging.info(f"Test R-squared = {model.score(X_test, Y_test):.4f}")

    Y_pred = model.predict(X_test)
    mae_test = np.mean(np.abs(Y_pred-Y_test), axis=0)

    logging.info(f"Test MAE = {mae_test:.4f} kcal/mol")

if __name__ == "__main__":
    main()


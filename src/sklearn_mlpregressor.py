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

import sklearn.neural_network

from grambow_dataset import grambow
from helpers import npz_ops

logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

SEED = 42

TEST_FRAC = 0.2

LAYER_SIZE = 200
LEARNING_RATE = 0.001
MAX_ITER = 100
N_ITER_NO_CHANGE = 10
TOL = 1e-6
VAL_FRAC = 0.1

RADIUS = 5
N_BITS = 1024

DATA_PATH = "../data/b97d3.csv"

def main() -> None:
    data_path = Path(DATA_PATH)

    # load data
    X_train, Y_train, X_test, Y_test = \
            grambow.load_data_2(data_path=data_path,
                                radius=RADIUS,
                                test_frac=TEST_FRAC,
                                n_bits=N_BITS)

    # training
    logging.info("Starting training...")
    model = sklearn.neural_network.MLPRegressor(
                hidden_layer_sizes=(LAYER_SIZE, LAYER_SIZE, ),
                learning_rate_init=LEARNING_RATE,
                random_state=SEED,
                early_stopping=True,
                validation_fraction=VAL_FRAC,
                max_iter=MAX_ITER,
                tol=TOL,
                n_iter_no_change=N_ITER_NO_CHANGE,
                verbose=True
            )
    model.fit(X_train, Y_train)

    logging.info(f"Finished training in {model.n_iter_} iterations.")
    logging.info(f"Test R-squared = {model.score(X_test, Y_test):.4f}")

    Y_pred = model.predict(X_test)
    mae_test = np.mean(np.abs(Y_pred-Y_test), axis=0)

    logging.info(f"Test MAE = {mae_test:.4f} kcal/mol")

    training_rmse = np.sqrt(
        np.asarray(model.loss_curve_, dtype=np.float32))
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(model.loss_curve_)+1),
             training_rmse, 
             color="r",
             label="Training loss")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(model.validation_scores_)+1),
             model.validation_scores_,
             color="k",
             label=r"Validation $R^2$ score")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(r"Training loss (RMSE, kcal/mol)")
    ax2.set_ylabel(r"$R^2$ score for validation")
    ax2.set_ylim(0.0, 1.0)
    fig.legend(ncols=2,
               bbox_to_anchor=(0.0, 1.005, 1.0, 0.005),
               loc="upper center")
    plt.show()

if __name__ == "__main__":
    main()


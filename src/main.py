"""Property predictor model"""
from pathlib import Path
from typing import Any, Optional
from pprint import pprint
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rdkit
import sklearn.neural_network
from rdkit.ML.Descriptors import MoleculeDescriptors

logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

SEED = 42

TEST_FRAC = 0.2

LAYER_SIZE = 200
LEARNING_RATE = 0.001
MAX_ITER = 40000
N_ITER_NO_CHANGE = 100
TOL = 1e-6
VAL_FRAC = 0.1

RADIUS = 5
N_BITS = 1024

DATA_PATH = Path("../data") / "b97d3.csv"
COMPRESSED_TRAIN_FEATURES_PATH = Path("../data") / "compressed" / "b97d3_X_train.npz"
COMPRESSED_TRAIN_TARGETS_PATH = Path("../data") / "compressed" / "b97d3_Y_train.npz"
COMPRESSED_TEST_FEATURES_PATH = Path("../data") / "compressed" / "b97d3_X_test.npz"
COMPRESSED_TEST_TARGETS_PATH = Path("../data") / "compressed" / "b97d3_Y_test.npz"


def compress_to_npz(data: np.ndarray[Any, Any], save_path: Path) -> None:
    """
    Compresses a numpy array to uncompressed npz format and saves to save_path.
    """
    np.savez(save_path, data=data)
    logging.info(f"Saved compressed array to {save_path}.")

def load_from_npz(npz_path: Path) -> np.ndarray[Any, Any]:
    """
    Load a compressed array from npz_path.
    """
    loaded_array = np.load(npz_path)
    logging.info(f"Loaded compressed array from {npz_path}.")
    return loaded_array["data"]

def convert_to_canonical(smile: str) -> str:
    """
    Given a SMILE string, returns its canonical representation.
    """
    molecule = rdkit.Chem.MolFromSmiles(smile)
    return rdkit.Chem.MolToSmiles(molecule)

def get_morgan_fingerprint(smile: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray[Any, Any]:
    """
    Generates Morgan fingerprint from SMILE string.
    """
    mol = rdkit.Chem.MolFromSmiles(smile)
    fingerprint = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.asarray(fingerprint, dtype=np.float32)

def generate_train_test_split_idxs(idxs: np.ndarray[Any, Any],
                                   test_frac: float,
                                   seed: int) -> \
                                    tuple[np.ndarray[Any, Any],
                                          np.ndarray[Any, Any]]:
    """
    Given a set of indices of data points, randomly splits indices
    into (test_frac * 100)% test indices and remaining training indices.
    """
    np.random.seed(seed)

    test_size = int(test_frac * idxs.shape[0])
    train_size = idxs.shape[0] - test_size
    perm_idxs = np.random.permutation(idxs)
    return perm_idxs[:train_size], perm_idxs[train_size:]

def generate_reverse_rxn_data(X_reactant: np.ndarray[Any, Any],
                              X_product: np.ndarray[Any, Any],
                              Y: np.ndarray[Any, Any],
                              delta_h: np.ndarray[Any, Any]) \
                            -> tuple[np.ndarray[Any, Any],
                                     np.ndarray[Any, Any]]:
    """
    Given a set of input features, reaction enthalpy and target variables
    for reactants and products in forward reactions, generates combined
    features and targets for corresponding reverse reactions.
    """
    X_diff = X_reactant - X_product
    Y = np.hstack([Y, Y-delta_h])
    delta_h = np.hstack([delta_h, -delta_h])
    X_diff = np.vstack([X_diff, -X_diff])
    X = np.hstack([X_diff, delta_h[:, np.newaxis]])

    return X, Y

def generate_features(data: pd.DataFrame, radius: int, n_bits: int) -> \
                   tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Encode reactant and product SMILES into Morgan fingerprints and
    return encoded reactant and product features
    """
    # feature engineering
    X_reactant = data["rsmi"].values
    X_product = data["psmi"].values
    
    X_reactant = data["rsmi"].apply(lambda x: get_morgan_fingerprint(x, radius=radius, n_bits=n_bits)).values
    X_product = data["psmi"].apply(lambda x: get_morgan_fingerprint(x, radius=radius, n_bits=n_bits)).values

    X_reactant = np.asarray(list(X_reactant), dtype=np.float32)
    X_product = np.asarray(list(X_product), dtype=np.float32)

    logging.info(f"Featurized {X_reactant.shape[0]} reactions into {X_reactant.shape[1]} features.")
    return X_reactant, X_product

def preprocess_training_data(data: pd.DataFrame,
                             radius: int,
                             n_bits: int,
                             compressed_features_path: Path,
                             compressed_targets_path: Path) -> None:
    """
    Given a raw training dataset, cleans, creates molecular fingerprints
    and saves processed features and targets in a compressed format.
    """
    logging.info("Preprocessing training data...")
    start = time.perf_counter()

    # making compressed data dir if it does not exist
    compressed_features_path.parent.mkdir(exist_ok=True, parents=True)

    # keeping only canonical smiles and removing duplicates
    rows = data.shape[0]
    data[["rsmi", "psmi"]] = data[["rsmi", "psmi"]].applymap(convert_to_canonical)
    data.drop_duplicates(subset=["rsmi", "psmi"], inplace=True)

    logging.info(f"Converted to canonical smiles. Dropped {rows-data.shape[0]} duplicate rows.")

    # defining target variable
    Y = data["ea"]

    # generating input features from canonical SMILES
    X_reactant, X_product = generate_features(data, radius, n_bits)

    # generating data for reverse reactions
    X, Y = generate_reverse_rxn_data(X_reactant=X_reactant,
                                     X_product=X_product,
                                     Y=Y,
                                     delta_h=data["dh"].values)

    # compressing data
    compress_to_npz(X, compressed_features_path)
    compress_to_npz(Y, compressed_targets_path)

    logging.debug(f"Reading and preprocessing took {time.perf_counter() - start} seconds.")

def preprocess_test_data(data: pd.DataFrame,
                         radius: int,
                         n_bits: int,
                         compressed_features_path: Path,
                         compressed_targets_path: Path) -> None:
    """
    Given a raw test dataset, cleans, creates molecular fingerprints
    and saves processed features and targets in a compressed format.
    """
    logging.info("Pre-processing test data...")
    start = time.perf_counter()

    # making compressed data dir if it does not exist
    compressed_features_path.parent.mkdir(exist_ok=True, parents=True)

    # keeping only canonical smiles and removing duplicates
    rows = data.shape[0]
    data[["rsmi", "psmi"]] = data[["rsmi", "psmi"]].applymap(convert_to_canonical)
    data.drop_duplicates(subset=["rsmi", "psmi"], inplace=True)

    logging.info(f"Converted to canonical smiles. Dropped {rows-data.shape[0]} duplicate rows.")

    # defining target variable
    Y = data["ea"]

    # generating input features from canonical SMILES
    X_reactant, X_product = generate_features(data, radius, n_bits)
    X = X_reactant - X_product
    delta_h = data["dh"].values[:, np.newaxis]
    X = np.hstack([X, delta_h])

    # compressing data
    compress_to_npz(X, compressed_features_path)
    compress_to_npz(Y, compressed_targets_path)

    logging.debug(f"Reading and preprocessing took {time.perf_counter() - start} seconds.")

def main() -> None:
    compressed_train_features_path = COMPRESSED_TRAIN_FEATURES_PATH
    compressed_train_targets_path = COMPRESSED_TRAIN_TARGETS_PATH
    compressed_test_features_path = COMPRESSED_TEST_FEATURES_PATH
    compressed_test_targets_path = COMPRESSED_TEST_TARGETS_PATH
    compressed_data_paths = [compressed_train_features_path,
                             compressed_train_targets_path,
                             compressed_test_features_path,
                             compressed_test_targets_path]

    data_path = DATA_PATH
    radius = RADIUS
    n_bits = N_BITS

    # load data

    if not all([path.exists() for path in compressed_data_paths]):
        # reading in data
        data = pd.read_csv(data_path, index_col="idx")
        logging.info(f"Read in data from {data_path}.")

        train_idxs, test_idxs = generate_train_test_split_idxs(
                                data.index.values, 
                                test_frac=TEST_FRAC,
                                seed=SEED)
        train_data = data.loc[train_idxs, :]
        test_data = data.loc[test_idxs, :]

        logging.info(f"Split dataset into {train_data.shape[0]} train and {test_data.shape[0]} test data points.")

        preprocess_training_data(train_data, 
                        radius=radius,
                        n_bits=n_bits,
                        compressed_features_path=compressed_train_features_path, 
                        compressed_targets_path=compressed_train_targets_path)
        preprocess_test_data(test_data, 
                        radius=radius,
                        n_bits=n_bits,
                        compressed_features_path=compressed_test_features_path, 
                        compressed_targets_path=compressed_test_targets_path)

    X_train, Y_train, X_test, Y_test = list(map(load_from_npz, compressed_data_paths))

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
    logging.info(f"Best loss = {model.best_loss_}.")
    logging.info(f"Accuracy = {model.score(X_test, Y_test)}")

    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(model.loss_curve_)+1),
             model.loss_curve_, 
             color="r",
             label="Training loss")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, len(model.validation_scores_)+1),
             model.validation_scores_,
             color="k",
             label=r"Validation $R^2$ score")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training loss")
    ax2.set_ylabel(r"$R^2$ score for validation")
    ax2.set_ylim(0.0, 1.0)
    fig.legend(ncols=2, loc="outside upper center")
    plt.show()

if __name__ == "__main__":
    main()


"""
The Grambow dataset and associated preprocessing functions.
"""
from typing import Any
from pathlib import Path
import logging
import time

import sys

sys.path.insert(0, str(Path(__file__).parents[1].resolve()))

import numpy as np
import pandas as pd

from helpers import preprocessing
from helpers import npz_ops

logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

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
    
    X_reactant = data["rsmi"].apply(lambda x: preprocessing.get_morgan_fingerprint(x, radius=radius, n_bits=n_bits)).values
    X_product = data["psmi"].apply(lambda x: preprocessing.get_morgan_fingerprint(x, radius=radius, n_bits=n_bits)).values

    X_reactant = np.asarray(list(X_reactant), dtype=np.float32)
    X_product = np.asarray(list(X_product), dtype=np.float32)

    logging.info(f"Featurized {X_reactant.shape[0]} reactions into {X_reactant.shape[1]} features.")
    return X_reactant, X_product

def preprocess_data(data: pd.DataFrame,
                    radius: int,
                    n_bits: int) -> \
                    tuple[np.ndarray[Any, Any], 
                          np.ndarray[Any, Any],
                          np.ndarray[Any, Any],
                          np.ndarray[Any, Any]]:
    """
    Given a raw training dataset, cleans, creates molecular fingerprints
    and returns reactant, product features, delta H and target arrays.
    """
    logging.info("Preprocessing training data...")
    start = time.perf_counter()

    # keeping only canonical smiles and removing duplicates
    rows = data.shape[0]
    data[["rsmi", "psmi"]] = data[["rsmi", "psmi"]].applymap(preprocessing.convert_to_canonical)
    data.drop_duplicates(subset=["rsmi", "psmi"], inplace=True)

    logging.info(f"Converted to canonical smiles. Dropped {rows-data.shape[0]} duplicate rows.")

    # defining target variable
    Y = data["ea"]
    
    # delta H array
    X_dh = data["dh"]

    # generating input features from canonical SMILES
    X_reactant, X_product = generate_features(data, radius, n_bits)

    logging.debug(f"Reading and preprocessing took {time.perf_counter() - start} seconds.")

    return X_reactant, X_product, X_dh, Y

def load_data_1(data_path: Path, 
              radius: int,
              n_bits: int,
              test_frac: float = 0.2,
              seed: int = 42)\
       -> tuple[np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                np.ndarray[Any, Any]]:
    """
    Load and preprocess training and test data from the Grambow dataset.
    This function generates training data that has both forward and 
    backward reactions, while the test data has only forward reactions.
    """

    compressed_train_features_path =  data_path.parent / "compressed" / "b97d3_X_train.npz"
    compressed_train_targets_path = data_path.parent / "compressed" / "b97d3_Y_train.npz"
    compressed_test_features_path = data_path.parent / "compressed" / "b97d3_X_test.npz"
    compressed_test_targets_path = data_path.parent / "compressed" / "b97d3_Y_test.npz"

    compressed_data_paths = [compressed_train_features_path,
                             compressed_train_targets_path,
                             compressed_test_features_path,
                             compressed_test_targets_path]

    if not all([path.exists() for path in compressed_data_paths]):
        data = pd.read_csv(data_path, index_col="idx")
        logging.info(f"Read in data from {data_path}.")

        train_idxs, test_idxs = generate_train_test_split_idxs(
                                data.index.values, 
                                test_frac=test_frac,
                                seed=seed)
        train_data = data.loc[train_idxs, :]
        test_data = data.loc[test_idxs, :]

        logging.info(f"Split dataset into {train_data.shape[0]} train and {test_data.shape[0]} test data points.")

        X_reactant_train, X_product_train, X_dh_train, Y_train = \
                preprocess_data(train_data, 
                                radius=radius,
                                n_bits=n_bits)
        X_reactant_test, X_product_test, X_dh_test, Y_test = \
                preprocess_data(test_data, 
                                radius=radius,
                                n_bits=n_bits)

        # generating data for reverse reactions for train data
        X_train, Y_train = generate_reverse_rxn_data(
                                     X_reactant=X_reactant_train,
                                     X_product=X_product_train,
                                     Y=Y_train,
                                     delta_h=X_dh_train)

        # generating input features for test data
        # from canonical SMILES without reverse reaction data
        X_test = X_reactant_test - X_product_test
        delta_h_test = X_dh_test[:, np.newaxis]
        X_test = np.hstack([X_test, delta_h_test])

        # compress to npz
        data_to_compress = [X_train, Y_train, X_test, Y_test]

        for data, path in zip(data_to_compress, compressed_data_paths):
            npz_ops.compress_to_npz(data, path)

    X_train, Y_train, X_test, Y_test = list(map(npz_ops.load_from_npz, 
                                                compressed_data_paths))
    return X_train, Y_train, X_test, Y_test

def load_data_2(data_path: Path, 
              radius: int,
              n_bits: int,
              test_frac: float = 0.2,
              seed: int = 42)\
       -> tuple[np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                np.ndarray[Any, Any]]:
    """
    Load and preprocess training and test data from the Grambow dataset.
    This function generates training and test data that have both forward 
    and backward reactions.
    """

    compressed_train_features_path =  data_path.parent / "compressed" / "b97d3_X_train.npz"
    compressed_train_targets_path = data_path.parent / "compressed" / "b97d3_Y_train.npz"
    compressed_test_features_path = data_path.parent / "compressed" / "b97d3_X_test.npz"
    compressed_test_targets_path = data_path.parent / "compressed" / "b97d3_Y_test.npz"

    compressed_data_paths = [compressed_train_features_path,
                             compressed_train_targets_path,
                             compressed_test_features_path,
                             compressed_test_targets_path]

    if not all([path.exists() for path in compressed_data_paths]):
        data = pd.read_csv(data_path, index_col="idx")
        logging.info(f"Read in data from {data_path}.")

        train_idxs, test_idxs = generate_train_test_split_idxs(
                                data.index.values, 
                                test_frac=test_frac,
                                seed=seed)
        train_data = data.loc[train_idxs, :]
        test_data = data.loc[test_idxs, :]

        logging.info(f"Split dataset into {train_data.shape[0]} train and {test_data.shape[0]} test data points.")

        X_reactant_train, X_product_train, X_dh_train, Y_train = \
                preprocess_data(train_data, 
                                radius=radius,
                                n_bits=n_bits)
        X_reactant_test, X_product_test, X_dh_test, Y_test = \
                preprocess_data(test_data, 
                                radius=radius,
                                n_bits=n_bits)

        # generating data for reverse reactions for train data
        X_train, Y_train = generate_reverse_rxn_data(
                                     X_reactant=X_reactant_train,
                                     X_product=X_product_train,
                                     Y=Y_train,
                                     delta_h=X_dh_train)

        # generating data for reverse reactions for train data
        X_test, Y_test = generate_reverse_rxn_data(
                                     X_reactant=X_reactant_test,
                                     X_product=X_product_test,
                                     Y=Y_test,
                                     delta_h=X_dh_test)

        # compress to npz
        data_to_compress = [X_train, Y_train, X_test, Y_test]

        for data, path in zip(data_to_compress, compressed_data_paths):
            npz_ops.compress_to_npz(data, path)

    X_train, Y_train, X_test, Y_test = list(map(npz_ops.load_from_npz, 
                                                compressed_data_paths))
    return X_train, Y_train, X_test, Y_test

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

from helpers import dataset, npz_ops, preprocessing

logging.basicConfig(
        format="%(asctime)s-%(levelname)s: %(message)s",
        level=logging.INFO)

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
    X_diff = X_product - X_reactant
    Y = np.hstack([Y, Y-delta_h])
    delta_h = np.hstack([delta_h, -delta_h])
    X_diff = np.vstack([X_diff, -X_diff])
    X = np.hstack([X_diff, delta_h[:, np.newaxis]])

    rng = np.random.default_rng()
    idxs = np.arange(X.shape[0])
    rng.shuffle(idxs)

    return X[idxs], Y[idxs]

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

def load_data_scaffold_split(
              data_path: Path, 
              radius: int,
              n_bits: int,
              val_frac: float,
              test_frac: float,
              data_url: str)\
       -> tuple[np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                np.ndarray[Any, Any],
                np.ndarray[Any, Any]]:
    """
    Load and preprocess training and test data from the Grambow dataset.
    This function generates data that has both forward and backward 
    reactions. A scaffold split of data based on reactant molecules
    is used.
    """

    compressed_train_features_path =  data_path.parent / "compressed" / "b97d3_scaffold_X_train.npz"
    compressed_train_targets_path = data_path.parent / "compressed" / "b97d3_scaffold_Y_train.npz"
    compressed_val_features_path =  data_path.parent / "compressed" / "b97d3_scaffold_X_val.npz"
    compressed_val_targets_path = data_path.parent / "compressed" / "b97d3_scaffold_Y_val.npz"
    compressed_test_features_path = data_path.parent / "compressed" / "b97d3_scaffold_X_test.npz"
    compressed_test_targets_path = data_path.parent / "compressed" / "b97d3_scaffold_Y_test.npz"

    compressed_data_paths = [compressed_train_features_path,
                             compressed_train_targets_path,
                             compressed_val_features_path,
                             compressed_val_targets_path,
                             compressed_test_features_path,
                             compressed_test_targets_path]

    if not all([path.exists() for path in compressed_data_paths]):
        if not data_path.exists():
            data_path.parent.mkdir(exist_ok=True)
            dataset.download_data(data_url, data_path.parent, data_path.name)
        data = pd.read_csv(data_path, index_col="idx")
        logging.info(f"Read in data from {data_path}.")

        train_idxs, val_idxs, test_idxs = \
                dataset.generate_scaffold_split_idxs(
                                data["rsmi"],
                                val_frac=val_frac,
                                test_frac=test_frac)
        train_data = data.loc[train_idxs, :]
        val_data = data.loc[val_idxs, :]
        test_data = data.loc[test_idxs, :]

        X_reactant_train, X_product_train, X_dh_train, Y_train = \
                preprocess_data(train_data, 
                                radius=radius,
                                n_bits=n_bits)
        X_reactant_val, X_product_val, X_dh_val, Y_val = \
                preprocess_data(val_data, 
                                radius=radius,
                                n_bits=n_bits)
        X_reactant_test, X_product_test, X_dh_test, Y_test = \
                preprocess_data(test_data, 
                                radius=radius,
                                n_bits=n_bits)

        # generating data for reverse reactions
        X_train, Y_train = generate_reverse_rxn_data(
                                     X_reactant=X_reactant_train,
                                     X_product=X_product_train,
                                     Y=Y_train,
                                     delta_h=X_dh_train)
        X_val, Y_val = generate_reverse_rxn_data(
                                     X_reactant=X_reactant_val,
                                     X_product=X_product_val,
                                     Y=Y_val,
                                     delta_h=X_dh_val)
        X_test, Y_test = generate_reverse_rxn_data(
                                     X_reactant=X_reactant_test,
                                     X_product=X_product_test,
                                     Y=Y_test,
                                     delta_h=X_dh_test)

        # compress to npz
        data_to_compress = [X_train, Y_train, X_val, Y_val, X_test, Y_test]

        # create compressed data dir if it does not exist
        compressed_data_paths[0].parent.mkdir(exist_ok=True, parents=True)

        for data, path in zip(data_to_compress, compressed_data_paths):
            npz_ops.compress_to_npz(data, path)

    X_train, Y_train, X_val, Y_val, X_test, Y_test \
            = list(map(npz_ops.load_from_npz, 
                        compressed_data_paths))
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_data_random_split(
              data_path: Path, 
              radius: int,
              n_bits: int,
              test_frac: float,
              seed: int,
              data_url: str)\
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
        if not data_path.exists():
            data_path.parent.mkdir(exist_ok=True)
            dataset.download_data(data_url, data_path.parent, data_path.name)

        data = pd.read_csv(data_path, index_col="idx")
        logging.info(f"Read in data from {data_path}.")

        train_idxs, test_idxs = dataset.generate_train_test_split_idxs(
                                data.index.values, 
                                test_frac=test_frac,
                                seed=seed)
        train_data = data.loc[train_idxs, :]
        test_data = data.loc[test_idxs, :]

        logging.info(f"Split dataset into {train_data.shape[0]} train "+
        f"and {test_data.shape[0]} test data points.")

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

        # create compressed data dir if it does not exist
        compressed_data_paths[0].parent.mkdir(exist_ok=True, parents=True)

        for data, path in zip(data_to_compress, compressed_data_paths):
            npz_ops.compress_to_npz(data, path)

    X_train, Y_train, X_test, Y_test = list(map(npz_ops.load_from_npz, 
                                                compressed_data_paths))
    return X_train, Y_train, X_test, Y_test

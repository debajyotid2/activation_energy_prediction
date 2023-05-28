"""
RGD1 dataset based on SMILES and Morgan fingerprints.
"""
import logging
from typing import Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import dataset, npz_ops, preprocessing

def _convert_smiles_to_canonical(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts reactant and product SMILE pairs to canonical SMILES
    using functions from RDKit. Removes rows containing duplicate
    pairs of reactants and products.
    """
    data[["reactant", "product"]] = data[["reactant", "product"]].applymap(preprocessing.convert_to_canonical)
    processed_data = data.drop_duplicates(subset=["reactant", "product"])
    percent_dropped = (data.shape[0] - processed_data.shape[0])/data.shape[0] * 100
    logging.info(f"Dropped data from {percent_dropped:.2f} % reactions.")
    return processed_data

def _generate_features(X_reactant: pd.Series,
                       X_product: pd.Series,
                       delta_h: pd.Series,
                       radius: int,
                       n_bits: int) \
                    -> np.ndarray[Any, Any]:
    """
    Converts SMILES to Morgan fingerprints using RDKit's functions. Calculates
    the difference between the reactant and product fingerprints and appends
    the un-normalized reaction enthalpy as an additional feature to get 
    the final feature vector for a single reaction. This process is repeated
    for all reactions.
    """
    X_reactant = X_reactant.apply(lambda x: 
                                     preprocessing.get_morgan_fingerprint(x, radius=radius, n_bits=n_bits))
    X_product = X_product.apply(lambda x: 
                                     preprocessing.get_morgan_fingerprint(x, radius=radius, n_bits=n_bits))
    X_reactant = np.asarray([reactant for reactant in X_reactant],
                            dtype=np.float32)
    X_product = np.asarray([product for product in X_reactant],
                           dtype=np.float32)
    
    diff = X_product - X_reactant
    return np.hstack([diff, delta_h.values[:, np.newaxis]])

def _compress_dataset(X: np.ndarray[Any, Any],
                      Y: np.ndarray[Any, Any],
                      compressed_data_dirpath: Path) -> None:
    """
    Compresses features and target arrays into npz format for
    fast I/O during training and inference.
    """
    compressed_data_dirpath.mkdir(exist_ok=True, parents=True)
    features_save_path = compressed_data_dirpath / "RGD1_features.npz"
    targets_save_path = compressed_data_dirpath / "RGD1_targets.npz"
    npz_ops.compress_to_npz(X, features_save_path)
    npz_ops.compress_to_npz(Y, targets_save_path)
   
def _load_compressed_dataset(compressed_data_dirpath: Path) \
                            -> tuple[Optional[np.ndarray[Any, Any]],
                                     Optional[np.ndarray[Any, Any]]]:
    """
    Loads features and targets from compressed npz files 
    saved in `compressed_data_dirpath`.
    """
    features_save_path = compressed_data_dirpath / "RGD1_features.npz"
    targets_save_path = compressed_data_dirpath / "RGD1_targets.npz"

    if not all([features_save_path.exists(), targets_save_path.exists()]):
        logging.warning(f"One or more compressed datasets not found in {compressed_data_dirpath}.")
        return None, None
    X, Y = npz_ops.load_from_npz(features_save_path), \
            npz_ops.load_from_npz(targets_save_path)

    return X.astype(np.float32), Y.astype(np.float32)

def load_data_random_split(data_path: Path,
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
    Loads the RGD1 dataset, generates Morgan fingerprints,
    compresses, splits dataset into train and test, and 
    loads transformed dataset into numpy arrays to return.
    """
    X, Y = _load_compressed_dataset(data_path.parent)

    if X is not None and Y is not None:
        train_idxs, test_idxs = dataset.generate_train_test_split_idxs(
                        np.arange(X.shape[0]), test_frac, seed)
        return X[train_idxs], Y[train_idxs], X[test_idxs], Y[test_idxs]

    if not data_path.exists():
        data_path.parent.mkdir(exist_ok=True)
        dataset.download_data(data_url, data_path.parent, data_path.name)
    _data = _convert_smiles_to_canonical(pd.read_csv(data_path))
    X = _generate_features(_data["reactant"],
                           _data["product"],
                           _data["DH"],
                           radius,
                           n_bits)
    _compress_dataset(X, _data["DE_F"].values, data_path.parent)
    X, Y = _load_compressed_dataset(data_path.parent)
    train_idxs, test_idxs = dataset.generate_train_test_split_idxs(
                        np.arange(X.shape[0]), test_frac, seed)
    return X[train_idxs], Y[train_idxs], X[test_idxs], Y[test_idxs]


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
    Splits the RGD1 dataset into train, validation and test sets,
    using Murcko scaffolds of reactant molecules.
    """
    X, Y = _load_compressed_dataset(data_path.parent)

    if X is not None and Y is not None:
        train_idxs, val_idxs, test_idxs = dataset.generate_scaffold_split_idxs(
                                    molecules=data["reactants"], 
                                    val_frac=val_frac, 
                                    test_frac=test_frac)
        return X[train_idxs], Y[train_idxs], X[val_idxs], Y[val_idxs],\
                        X[test_idxs], Y[test_idxs]
    
    if not data_path.exists():
        data_path.parent.mkdir(exist_ok=True)
        dataset.download_data(data_url, data_path.parent, data_path.name)
    data = _convert_smiles_to_canonical(pd.read_csv(data_path))
    
    X = _generate_features(data["reactant"],
                           data["product"],
                           data["DH"],
                           radius,
                           n_bits)
    _compress_dataset(X, _data["DE_F"].values, data_path.parent)
    X, Y = _load_compressed_dataset(data_path.parent)
        
    return X[train_idxs], Y[train_idxs], X[test_idxs], Y[test_idxs]


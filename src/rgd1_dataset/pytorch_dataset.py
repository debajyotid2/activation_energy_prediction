"""
Custom PyTorch dataset for the RGD1 reaction dataset. (https://figshare.com/articles/dataset/model_reaction_database/21066901)
"""
from pathlib import Path
from typing import Any, Optional
import logging
import urllib.request
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # enables relative imports
from helpers import preprocessing, npz_ops

logging.basicConfig(format="%(asctime)s-%(level)s: %(message)s",
                    level=logging.DEBUG)

class RGD1(torch.utils.data.Dataset):
    _url = "https://figshare.com/ndownloader/files/38170326"

    """
    Custom PyTorch dataset for the RGD1 reaction dataset. 
    Original dataset authored by Qiyuan Zhao, Brett Savoie, 
    Michael Woulfe, Sai Mahit Vaddadi, Lawal A. Ogunfowora,
    Sanjay Garimella.
    """
    def __init__(self, 
                 download_dirpath: Path,
                 radius: int,
                 n_bits: int):
        super().__init__()

        download_dirpath.resolve().mkdir(exist_ok=True, parents=True)

        self._X, self._Y = self._load_compressed_dataset(download_dirpath)
        if all([self._X is None, self._Y is None]):
            download_path = download_dirpath / "RGD1.csv"
            
            if not download_path.exists():
                self._download(self._url, download_path)
            _data = self._convert_smiles_to_canonical(pd.read_csv(download_path))
            X = self._generate_features(_data["reactant"],
                                        _data["product"],
                                        _data["DH"],
                                        radius,
                                        n_bits)
            self._compress_dataset(X, _data["DE_F"].values, download_dirpath)
            self._X, self._Y = self._load_compressed_dataset(download_dirpath)
        
    def _download(self, url: str, download_path: Path) -> None:
        """
        Downloads the CSV dataset from Figshare.
        """
        logging.info("Starting download...")
        urllib.request.urlretrieve(self._url, download_path)
        logging.info(f"Downloaded from {url} to {download_path}.")

    def _convert_smiles_to_canonical(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def _generate_features(self,
                           X_reactant: pd.Series,
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
    
    def _compress_dataset(self,
                          X: np.ndarray[Any, Any],
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
       
    def _load_compressed_dataset(self,
                                 compressed_data_dirpath: Path) \
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

    def __getitem__(self, idx: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        X = torch.from_numpy(self._X[idx, :])
        Y = torch.tensor(self._Y[idx], dtype=torch.float32)

        return X, Y 

    def __len__(self) -> int:
        return self._X.shape[0]

def load_dataloaders(download_dirpath: Path,
              radius: int,
              n_bits: int, 
              batch_size: int = 32,
              val_frac: float = 0.1,
              test_frac: float = 0.1,
              num_workers: int = 4,
              seed: int = 42) -> \
            tuple[torch.utils.data.DataLoader,
                  torch.utils.data.DataLoader,
                  torch.utils.data.DataLoader]:
    """
    Splits the RGD1 dataset into train, validation and test sets,
    loads them into respective dataloaders and returns the dataloaders.
    """
    train_frac = 1-(val_frac+test_frac)

    dataset = RGD1(download_dirpath=download_dirpath,
                   radius=radius,
                   n_bits=n_bits)

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
                                dataset,
                                [train_frac, val_frac, test_frac],
                                generator=torch.Generator().manual_seed(seed))
    train_loader = torch.utils.data.DataLoader(train_ds, 
                                       batch_size=batch_size, 
                                       shuffle=True,
                                       num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(val_ds, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       num_workers=num_workers)
    return train_loader, val_loader, test_loader

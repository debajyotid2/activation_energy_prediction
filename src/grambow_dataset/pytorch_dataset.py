"""
torch.utils.data.Dataset and torch.utils.data.DataLoader for the Grambow dataset.
"""
import os
import logging

from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import grambow
from helpers import dataset

logging.basicConfig(format="%(asctime)s-%(levelname)s: %(message)s",
                    level=logging.DEBUG)

URL = "https://zenodo.org/record/3715478/files/b97d3.csv?download=1"

class GrambowDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for the Grambow dataset (Colin A., Lagnajit Pattanaik,
    and William H. Green. "Reactants, products, and transition states of 
    elementary chemical reactions based on quantum chemistry." Scientific
    data 7.1 (2020): 137.)
    """
    def __init__(self,
                 X: np.ndarray[Any, Any],
                 Y: np.ndarray[Any, Any]):
        assert X.shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y

    def __getitem__(self, idx: int | list[int]) \
            -> tuple[torch.Tensor, torch.Tensor]:
        X_points = np.asarray(self.X[idx], dtype=np.float32)
        return torch.from_numpy(X_points),  \
                torch.tensor(self.Y[idx], dtype=torch.float32)

    def __len__(self) -> int:
        return self.Y.shape[0]

def load_dataloaders_scaffold_split(
                   data_dirpath: Path,
                   radius: int,
                   n_bits: int,
                   batch_size: int,
                   val_frac: float,
                   test_frac: float,
                   num_workers: int,
                   data_url: str) \
                  -> tuple[torch.utils.data.DataLoader,
                           torch.utils.data.DataLoader,
                           torch.utils.data.DataLoader]:
    """
    Loads training, validation and test dataloaders for
    the Grambow dataset with data split according to Bemis-Murcko
    scaffolds of the reactant molecules across the subsets. 
    """
    num_workers = os.cpu_count()-1 if not num_workers else num_workers
    
    data_dirpath.mkdir(exist_ok=True, parents=True)
    dataset.download_data(data_url, data_dirpath, "b97d3.csv")
    csvpath = data_dirpath / "b97d3.csv"
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
            grambow.load_data_scaffold_split(
                                          data_path=csvpath,
                                          radius=radius,
                                          n_bits=n_bits,
                                          val_frac=val_frac,
                                          test_frac=test_frac,
                                          data_url=data_url)

    train_ds = GrambowDataset(X_train, Y_train)
    val_ds = GrambowDataset(X_val, Y_val)
    test_ds = GrambowDataset(X_test, Y_test)

    train_dataloader = torch.utils.data.DataLoader(
                dataset=train_ds, 
                batch_size=batch_size,
                num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(
                dataset=val_ds, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(
                dataset=test_ds, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers)

    logging.info(f"Train points: {len(X_train)}, validation points: {len(X_val)}, test points: {len(X_test)}.")
    return train_dataloader, val_dataloader, test_dataloader

def load_dataloaders_random_split(
                   data_dirpath: Path,
                   radius: int,
                   n_bits: int,
                   batch_size: int,
                   val_frac: float,
                   test_frac: float,
                   num_workers: int,
                   seed: int,
                   data_url: str) \
                  -> tuple[torch.utils.data.DataLoader,
                           torch.utils.data.DataLoader,
                           torch.utils.data.DataLoader]:
    """
    Loads training, validation and test dataloaders for
    the Grambow dataset with data randomly split across
    the subsets. The loading_func decides the presence of 
    reverse reaction data in each subset.
    """
    num_workers = os.cpu_count()-1 if not num_workers else num_workers
    
    data_dirpath.mkdir(exist_ok=True, parents=True)
    dataset.download_data(data_url, data_dirpath, "b97d3.csv")
    csvpath = data_dirpath / "b97d3.csv"
    X_train, Y_train, X_test, Y_test = grambow.load_data_random_split(
                                          data_path=csvpath,
                                          radius=radius,
                                          n_bits=n_bits,
                                          test_frac=test_frac,
                                          seed=seed,
                                          data_url=data_url)

    train_ds = GrambowDataset(X_train, Y_train)
    test_ds = GrambowDataset(X_test, Y_test)

    train_idxs, val_idxs = dataset.generate_train_test_split_idxs(
                np.arange(X_train.shape[0]), val_frac, seed
            )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampler = torch.utils.data.SubsetRandomSampler(
                indices=train_idxs, generator=generator
            )
    val_sampler = torch.utils.data.SubsetRandomSampler(
                indices=val_idxs, generator=generator
            )

    train_dataloader = torch.utils.data.DataLoader(
                dataset=train_ds, 
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(
                dataset=train_ds, 
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(
                dataset=test_ds, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers)

    logging.info(f"Train points: {len(train_idxs)}, validation points: {len(val_idxs)}, test points: {len(test_ds)}.")
    return train_dataloader, val_dataloader, test_dataloader

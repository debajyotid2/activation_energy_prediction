"""
torch.utils.data.Dataset and torch.utils.data.DataLoader for the Grambow dataset.
"""
import os
import logging

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from . import download, grambow

LoadingFunc = Callable[[Path, int, int, float, int],\
                        tuple[np.ndarray[Any, Any],
                              np.ndarray[Any, Any],
                              np.ndarray[Any, Any],
                              np.ndarray[Any, Any]]]

logging.basicConfig(format="%(asctime)s-%(levelname)s: %(message)s",
                    level=logging.DEBUG)

class GrambowDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for the Grambow dataset (Colin A., Lagnajit Pattanaik,
    and William H. Green. "Reactants, products, and transition states of 
    elementary chemical reactions based on quantum chemistry." Scientific
    data 7.1 (2020): 137.)
    """
    __url = "https://zenodo.org/record/3715478/files/b97d3.csv?download=1"

    def __init__(self,
                 data_dirpath: Path,
                 radius: int,
                 n_bits: int,
                 test: bool = False,
                 test_frac: float = 0.2,
                 dataset_frac: float = 1.0,
                 loading_function: LoadingFunc = grambow.load_data_1,
                 seed: int = 42):
        data_dirpath.mkdir(exist_ok=True, parents=True)
        download.download_data(self.__url, data_dirpath)
        csvpath = data_dirpath / "b97d3.csv"
        X_train, Y_train, X_test, Y_test = loading_function(
                                              data_path=csvpath,
                                              radius=radius,
                                              n_bits=n_bits,
                                              test_frac=test_frac,
                                              seed=seed)
        self.X = X_test[:int(dataset_frac*X_test.shape[0])]\
                        if test else \
                        X_train[:int(dataset_frac*X_train.shape[0])]
        self.Y = Y_test[:int(dataset_frac*Y_test.shape[0])]\
                        if test else \
                        Y_train[:int(dataset_frac*X_train.shape[0])]

    def __getitem__(self, idx: int | list[int]) \
            -> tuple[torch.Tensor, torch.Tensor]:
        X_points = np.asarray(self.X[idx], dtype=np.float32)
        return torch.from_numpy(X_points),  \
                torch.tensor(self.Y[idx], dtype=torch.float32)

    def __len__(self) -> int:
        return self.Y.shape[0]

def load_dataloaders_1(
                   data_dirpath: Path,
                   radius: int,
                   n_bits: int,
                   batch_size: int = 32,
                   val_frac: float = 0.2,
                   test_frac: float = 0.2,
                   num_workers: int = 4,
                   dataset_frac: float = 1.0,
                   seed: int = 42) \
                  -> tuple[torch.utils.data.DataLoader,
                           torch.utils.data.DataLoader,
                           torch.utils.data.DataLoader]:
    """
    Loads training, validation and test dataloaders for
    the Grambow dataset. The training and validation datasets
    have both forward and backward reactions, while the test
    dataset has only the forward reactions.
    """
    num_workers = os.cpu_count()-1 if not num_workers else num_workers

    train_ds = GrambowDataset(data_dirpath=data_dirpath,
                             radius=radius,
                             n_bits=n_bits,
                             test_frac=test_frac,
                             test=False,
                             dataset_frac=dataset_frac,
                             seed=seed)
    test_ds = GrambowDataset(data_dirpath=data_dirpath,
                             radius=radius,
                             n_bits=n_bits,
                             test=True,
                             test_frac=test_frac,
                             dataset_frac=dataset_frac,
                             seed=seed)

    train_idxs = np.random.default_rng(seed=seed)\
                          .permutation(np.arange(len(train_ds)))
    val_idxs = train_idxs[:int(val_frac*len(train_ds))]
    train_idxs = train_idxs[int(val_frac*len(train_ds)):]
    
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

def load_dataloaders_2(
                   data_dirpath: Path,
                   radius: int,
                   n_bits: int,
                   batch_size: int = 32,
                   val_frac: float = 0.2,
                   test_frac: float = 0.2,
                   num_workers: int = 4,
                   dataset_frac: float = 1.0,
                   seed: int = 42) \
                  -> tuple[torch.utils.data.DataLoader,
                           torch.utils.data.DataLoader,
                           torch.utils.data.DataLoader]:
    """
    Loads training, validation and test dataloaders for
    the Grambow dataset. The training, validation and test datasets
    have both forward and backward reactions.
    """
    num_workers = os.cpu_count()-1 if not num_workers else num_workers

    train_ds = GrambowDataset(data_dirpath=data_dirpath,
                             radius=radius,
                             n_bits=n_bits,
                             test_frac=test_frac,
                             test=False,
                             dataset_frac=dataset_frac,
                             loading_function=grambow.load_data_2,
                             seed=seed)
    test_ds = GrambowDataset(data_dirpath=data_dirpath,
                             radius=radius,
                             n_bits=n_bits,
                             test=True,
                             test_frac=test_frac,
                             dataset_frac=dataset_frac,
                             loading_function=grambow.load_data_2,
                             seed=seed)

    train_idxs = np.random.default_rng(seed=seed)\
                          .permutation(np.arange(len(train_ds)))
    val_idxs = train_idxs[:int(val_frac*len(train_ds))]
    train_idxs = train_idxs[int(val_frac*len(train_ds)):]
    
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


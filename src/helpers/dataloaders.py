"""
Module containing functions that organize imports into
caller functions that return the appropriate data loading
function given the type of dataset and data split.
"""
from typing import Any, Callable
from functools import partial
from pathlib import Path
import numpy as np
import torch
from grambow_dataset import grambow
from rgd1_dataset import rgd1
from grambow_dataset import pytorch_dataset as grambow_pytorch_dataset
from rgd1_dataset import pytorch_dataset as rgd1_pytorch_dataset

PandasLoadingFunc = Callable[[Path, int, float, float, int],
                              tuple[np.ndarray[Any, Any], 
                                    np.ndarray[Any, Any],
                                    np.ndarray[Any, Any], 
                                    np.ndarray[Any, Any]]]
PytorchLoadingFunc = Callable[[Path, int, int, int, float,
                               float, int, float, int],
                            tuple[torch.utils.data.DataLoader,
                                  torch.utils.data.DataLoader,
                                  torch.utils.data.DataLoader]]

def scaffold_split_loader(loader_func: Callable[Any, Any],
                          data_path: Path,
                          radius: int,
                          n_bits: int,
                          val_frac: float,
                          test_frac: float,
                          data_url: str) ->\
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any],
                  np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Placeholder function that uses the scaffold split loader function
    to load the data split according to scaffolds and merge validation
    and training splits into single training splits.
    """
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
                    loader_func(data_path=data_path,
                                radius=radius,
                                n_bits=n_bits,
                                val_frac=val_frac,
                                test_frac=test_frac,
                                data_url=data_url)
    X_train = np.vstack([X_train, X_val])
    Y_train = np.hstack([Y_train, Y_val])
    
    return X_train, Y_train, X_val, Y_val

PYTORCH_DATALOADERS = \
{
    ("grambow", "random"): \
            grambow_pytorch_dataset.load_dataloaders_random_split,
    ("grambow", "scaffold"): \
            grambow_pytorch_dataset.load_dataloaders_scaffold_split, 
    ("rgd1", "random"): rgd1_pytorch_dataset.load_dataloaders_random_split,
    ("rgd1", "scaffold"): rgd1_pytorch_dataset.load_dataloaders_scaffold_split
}

PANDAS_DATALOADERS = \
{
    ("grambow", "random"): grambow.load_data_random_split,
    ("grambow", "scaffold"): partial(scaffold_split_loader, 
                                     grambow.load_data_scaffold_split),
    ("rgd1", "random"): rgd1.load_data_random_split,
    ("rgd1", "scaffold"): partial(scaffold_split_loader,
                                  rgd1.load_data_scaffold_split)
}

def get_pandas_dataloader(dataset: str,
                   split_type: str) -> PandasLoadingFunc:
    """
    Returns the appropriate pandas data loading function given the 
    dataset and type of data split used.
    """
    if (dataset, split_type) not in PANDAS_DATALOADERS:
        raise ValueError(f"No loading function found for {dataset} and {split_type}.")
    return PANDAS_DATALOADERS[(dataset, split_type)]

def get_pytorch_dataloader(dataset: str,
                   split_type: str) -> PytorchLoadingFunc:
    """
    Returns the appropriate pytorch data loading function given the 
    dataset and type of data split used.
    """
    if (dataset, split_type) not in PYTORCH_DATALOADERS:
        raise ValueError(f"No loading function found for {dataset} and {split_type}.")
    return PYTORCH_DATALOADERS[(dataset, split_type)]

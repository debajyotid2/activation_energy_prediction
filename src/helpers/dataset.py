"""
Miscellaneous helper functions for preparing datasets.
"""
import urllib.request
import logging
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit.Chem.Scaffolds.MurckoScaffold

logging.basicConfig(format="%(asctime)s-%(levelname)s: %(message)s",
                    level=logging.DEBUG)

def download_data(url: str, data_dir: Path, filename: str) -> None:
    """
    Downloads files from dataset into specified data directory.
    """
    filepath = data_dir / filename
    
    if not filepath.exists():
        logging.info(f"Downloading from {url} ...")
        urllib.request.urlretrieve(url, filepath) 
        logging.info(f"{filepath} downloaded.")
    else:
        logging.warning(f"{filepath} already exists.")

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

def generate_scaffolds(smiles: pd.Series,
                           include_chirality: bool = False)\
                        -> pd.DataFrame:
    """
    Returns a dataframe of Bemis-Murcko scaffolds generated from
    supplied pandas series of SMILES strings.
    """
    smiles.name = "smiles"
    scaffolds = smiles.to_frame()
    scaffolds["scaffold"] = scaffolds["smiles"].apply(
            lambda x: rdkit.Chem.Scaffolds.MurckoScaffold\
                      .MurckoScaffoldSmiles(
                            x, includeChirality=include_chirality))
    return scaffolds

def generate_mask_by_scaffold(scaffolds: pd.DataFrame,
                              subset_scaffold_counts: pd.DataFrame,
                              subset_fracs: tuple[float] | tuple[float,float]\
                                      | tuple[float|float|float],
                              size_limit: int = 0,
                              offset: int = 0) \
                            -> tuple[np.ndarray[Any, Any],
                                     np.ndarray[Any, Any]]:
    """
    Generates a mask of length equal to the dataset
    with the indices belonging to a subset of scaffold
    split marked "True". Returns this mask along with
    a mask over the subset_scaffold_counts indicating 
    which scaffolds were used in this assignment.
    """
    subset_len = (scaffolds.shape[0] - offset) * max(subset_fracs)
    subset_mask = scaffolds["scaffold"].isnull()
    subset_scaffold_mask = ((subset_scaffold_counts > size_limit) & 
                            (subset_scaffold_counts.cumsum() <= subset_len))
    subset_scaffolds = subset_scaffold_counts[subset_scaffold_mask]\
                                        .index.tolist()

    for scaffold in subset_scaffolds:
        subset_mask |= (scaffolds["scaffold"] == scaffold)

    if subset_mask.sum() < subset_len:
        rem_subset_scaffold_mask = ~subset_scaffold_mask & \
                (subset_scaffold_counts.cumsum() <= subset_len)
        subset_scaffolds = subset_scaffold_counts[rem_subset_scaffold_mask]\
                                            .index.tolist()
        for scaffold in subset_scaffolds:
            subset_mask |= (scaffolds["scaffold"] == scaffold)
        subset_scaffold_mask |= rem_subset_scaffold_mask
    return subset_mask, subset_scaffold_mask

def generate_scaffold_split_idxs(
            molecules: pd.Series,
            val_frac: float = 0.1,
            test_frac: float = 0.1,
            include_chirality: bool = False,
            seed: int = 42
        ) \
                -> tuple[np.ndarray[Any, Any],
                         np.ndarray[Any, Any],
                         np.ndarray[Any, Any]]:
    """
    Given a pandas series of reactant or product molecules as 
    SMILES strings and test and validation fractions, splits
    the indices according to splits based on Bemis-Murcko scaffolds
    of the molecules. Returns train, validation and test indices.
    """
    if val_frac+test_frac > 0.5:
        raise ValueError(
            "Validation and test sets cannot be larger"+
            "than half the dataset.")

    train_frac = 1 - (test_frac + val_frac)
    scaffolds = generate_scaffolds(molecules, include_chirality)

    scaffold_counts = scaffolds["scaffold"].value_counts()
    
    # Leave out points with no scaffold assigned ("orphans")
    offset = scaffold_counts[""]
    scaffold_counts = scaffold_counts[scaffold_counts.index != ""]

    limit = (molecules.shape[0] - offset) * min(val_frac, test_frac) / 2
    train_mask, train_scaffold_mask = generate_mask_by_scaffold(
                        scaffolds, scaffold_counts, 
                        (train_frac, val_frac, test_frac), limit, offset)
        
    # Split remaining scaffolds between subset_1 and subset_2
    subset_1_scaffold_counts = scaffold_counts[~train_scaffold_mask]
    subset_1_mask, subset_1_scaffold_mask = generate_mask_by_scaffold(
                        scaffolds, subset_1_scaffold_counts, 
                        (val_frac, test_frac), 0, offset)
    
    subset_2_scaffold_counts = subset_1_scaffold_counts[
                                        ~subset_1_scaffold_mask]
    subset_2_mask, subset_2_scaffold_mask = generate_mask_by_scaffold(
                        scaffolds, subset_2_scaffold_counts, 
                        (min(val_frac, test_frac),), 0, offset)
    
    # Add any remaining scaffolds to train mask, subset 1 mask and
    # subset 2 mask, in that order
    subset_1_frac = max(val_frac, test_frac)
    subset_2_frac = min(val_frac, test_frac)

    rem_scaffolds = subset_2_scaffold_counts[~subset_2_scaffold_mask]
    count = 0
    train_len = (molecules.shape[0]-offset+rem_scaffolds.sum())\
                                    * train_frac
    subset_1_len = (molecules.shape[0]-offset+rem_scaffolds.sum())\
                                    * subset_1_frac
    subset_2_len = (molecules.shape[0]-offset+rem_scaffolds.sum())\
                                    * subset_2_frac

    train_scaf_count = train_scaffold_mask.sum()
    subset_1_scaf_count = subset_1_scaffold_mask.sum()
    subset_2_scaf_count = subset_2_scaffold_mask.sum()
    
    while train_mask.sum() < train_len and \
            count < rem_scaffolds.shape[0]:
        train_mask |= (scaffolds["scaffold"] == 
                       rem_scaffolds.index[count])
        count += 1
        train_scaf_count += 1
    while subset_1_mask.sum() < subset_1_len and \
            count < rem_scaffolds.shape[0]:
        subset_1_mask |= (scaffolds["scaffold"] == 
                          rem_scaffolds.index[count])
        count += 1
        subset_1_scaf_count += 1
    while subset_2_mask.sum() < subset_2_len and \
            count < rem_scaffolds.shape[0]:
        subset_2_mask |= (scaffolds["scaffold"] == 
                          rem_scaffolds.index[count])
        count += 1
        subset_2_scaf_count += 1

    # Convert masks to indices
    all_idxs = np.arange(molecules.shape[0])
    train_idxs = all_idxs[train_mask]
    subset_1_idxs = all_idxs[subset_1_mask]
    subset_2_idxs = all_idxs[subset_2_mask]

    # Assign "orphan" molecules randomly to train, subset_1 and subset_2
    orphan_idxs = scaffolds[scaffolds["scaffold"] == ""].index.to_numpy()
    
    subset_1_len = int(molecules.shape[0] * subset_1_frac)
    subset_2_len = int(molecules.shape[0] * subset_2_frac)
    
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(orphan_idxs)
    
    # Filling in increasing order of subset length
    subset_2_idxs = np.hstack([subset_2_idxs, 
                       orphan_idxs[:subset_2_len]])
    subset_1_idxs = np.hstack([subset_1_idxs,
                   orphan_idxs[subset_2_len:subset_1_len+subset_2_len]])
    train_idxs = np.hstack([train_idxs, 
                    orphan_idxs[subset_1_len+subset_2_len:]])

    logging.debug(f"Total scaffolds: {scaffold_counts.shape[0]}.")
    logging.debug("Scaffold assignments:")
    logging.debug(f"Train: {train_scaf_count}")
    logging.debug(f"Validation: {subset_1_scaf_count if val_frac > test_frac else subset_2_scaf_count}")
    logging.debug(f"Test: {subset_2_scaf_count if val_frac > test_frac else subset_1_scaf_count}")

    val_idxs = subset_1_idxs if val_frac > test_frac\
                    else subset_2_idxs
    test_idxs = subset_2_idxs if val_frac > test_frac\
                    else subset_1_idxs

    logging.debug(f"Number of points: total: {molecules.shape[0]}, "+\
            f"train: {train_idxs.shape[0]}, validation: "+\
            f"{val_idxs.shape[0]}, test: {test_idxs.shape[0]}.")
    return train_idxs, val_idxs, test_idxs


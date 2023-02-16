"""Property predictor model"""
import re
from pathlib import Path
from typing import Optional
import urllib.request
import logging

import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit.ML.Descriptors import MoleculeDescriptors

SEED = 42

LAYER_SIZE = 200
LEARNING_RATE = 0.001

RADIUS = 5
N_BITS = 1024

DATA_DIR = Path("../data")

def get_morgan_fingerprints(smile: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray[Any, Any]:
    """
    Generates Morgan fingerprints from SMILE strings.
    """
    mol = rdkit.Chem.MolFromSmiles(smile)
    fingerprints = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.asarray(fingerprints, dtype=np.float32)
    
def main() -> None:
    data_dir = DATA_DIR
    

    classifier = sklearn.neural_network.MLPClassifier(
                hidden_layer_sizes=(LAYER_SIZE, LAYER_SIZE,),
                learning_rate=LEARNING_RATE,
                random_state=SEED
            )

if __name__ == "__main__":
    main()


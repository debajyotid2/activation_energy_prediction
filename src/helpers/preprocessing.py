"""
Functions for converting SMILES to fingerprints using RDKit.
"""

from typing import Any
import numpy as np
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors

def get_morgan_fingerprint(smile: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray[Any, Any]:
    """
    Generates Morgan fingerprint from SMILE string.
    """
    mol = rdkit.Chem.MolFromSmiles(smile)
    fingerprint = rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits)
    return np.asarray(fingerprint, dtype=np.float32)

def convert_to_canonical(smile: str) -> str:
    """
    Given a SMILE string, returns its canonical representation.
    """
    molecule = rdkit.Chem.MolFromSmiles(smile)
    return rdkit.Chem.MolToSmiles(molecule)



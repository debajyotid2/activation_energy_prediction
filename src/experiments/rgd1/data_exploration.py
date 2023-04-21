"""
Exploratory analysis on the Grambow dataset.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from grambow_dataset import grambow
from helpers import npz_ops, dataset

DATA_PATH = "../data/b97d3.csv"
COMPRESSED_DATA_DIR = "../data/temp_compressed"
RADIUS = 5
N_BITS = 1024

def plot_PCA(data: np.ndarray[Any, Any]) -> None:
    """
    Given a matrix, generate PCA plot along 2 components.
    """
    data = (data-np.mean(data, axis=0))/np.std(data, axis=0)

    pca_data = PCA(n_components=2).fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1])
    plt.show()
    plt.close()

def main() -> None:
    # Read and preprocess data
    data = pd.read_csv(DATA_PATH, index_col="idx")

    # X_r_path = Path(COMPRESSED_DATA_DIR)/"X_r.npz"
    # X_p_path = Path(COMPRESSED_DATA_DIR)/"X_p.npz"
    # X_dh_path = Path(COMPRESSED_DATA_DIR)/"X_dh.npz"
    # Y_f_path = Path(COMPRESSED_DATA_DIR)/"Y_fwd.npz"

    # data_paths = [X_r_path, X_p_path, X_dh_path, Y_f_path]

    # if not all([path.exists() for path in data_paths]):
    #     X_reactant, X_product, X_dh, Y_fwd = grambow.preprocess_data(data,
    #                                                        radius=RADIUS,
    #                                                        n_bits=N_BITS)
    #     npz_ops.compress_to_npz(X_reactant, X_r_path)
    #     npz_ops.compress_to_npz(X_product, X_p_path)
    #     npz_ops.compress_to_npz(X_dh, X_dh_path)
    #     npz_ops.compress_to_npz(Y_fwd, Y_f_path)

    # X_reactant = npz_ops.load_from_npz(X_r_path)
    # X_product = npz_ops.load_from_npz(X_p_path)
    # X_dh = npz_ops.load_from_npz(X_dh_path)
    # Y_fwd = npz_ops.load_from_npz(Y_f_path)
    # 
    # X_fwd = X_product - X_reactant
    # X_fwd = np.hstack([X_fwd, X_dh.reshape(-1, 1)])

    # X_rev = X_reactant - X_product
    # X_rev = np.hstack([X_rev, -X_dh.reshape(-1, 1)])

    # Y_rev = Y_fwd - X_dh

    dataset.generate_scaffold_split_idxs(
            data["rsmi"], 0.1, 0.2)

    # ea_df = pd.DataFrame(np.vstack([Y_fwd, Y_rev]).T, 
    #                      columns=["forward", "reverse"])
    
    # Correlation between activation energy of forward
    # and reverse reactions
    # g = sns.displot(data=ea_df, x="forward", y="reverse",
    #             rug=True, height=6, aspect=8/6)
    # g.set_xlabels(size=12)
    # g.set_ylabels(size=12)
    # g.set_xticklabels(size=12)
    # g.set_yticklabels(size=12)
    # plt.title("Activation energy of forward versus reverse reactions",
    #           fontsize="x-large")
    # plt.show()
    
    # Compare features of forward against backward reactions
    # plot_PCA(X_fwd)
    # plot_PCA(X_rev)

if __name__=="__main__":
    main()

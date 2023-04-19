# Notes from MLP regressor runs on Grambow's dataset

- only forward reactions in the test dataset
- both forward and backward reactions in the test dataset
- normalize $\Delta$ H before using it as a feature
- use $\Delta$ H directly without normalization

2 x 2 -> 4 runs

# Notes from Grambow's paper on the dataset

- reactions in the dataset consist of at most seven heavy atoms (C, N, O)
- for a single reactant chosen, a reaction network was grown around it.
- the dataset can be considered to comprise of a handful of reaction networks

# Experiments with Grambow dataset

val = 0.1, test = 0.2, seed = 243

## Forward only vs forward + backward in test set

### Forward only
Regressor Test loss Test MAE Test R^2

MLP 302.69 12.97 0.3903

### Forward and backward
Regressor Test loss Test MAE Test R^2

MLP 154.70 7.78 0.8291

## Murcko scaffold split
Regressor Test loss Test MAE Test R^2

MLP 440.63 16.41 0.5717

# Experiments with RGD1 dataset

## Random split
Regressor Test loss Test MAE Test R^2

MLP 609.81 19.54 0.2697

### Murcko scaffold split
Regressor Test loss Test MAE Test R^2

MLP 440.63 16.41 0.5717

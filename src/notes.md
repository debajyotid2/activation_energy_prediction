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
learning rate = 0.001, epochs = 100

## Forward only vs forward + backward in test set

### Forward only
Regressor Test loss Test MAE Test R^2

MLP 302.69 12.97 0.3903
Linear 395.16 16.04 0.2008
Ridge 395.16 16.05 0.2008
Lasso 395.04 16.04 0.2010
Random forest 219.08 10.52 0.5569 
Gradient boosted random forest 232.34 11.30 0.5301

### Forward and backward
Regressor Test loss Test MAE Test R^2

MLP 292.91 12.90 0.6757
Linear 395.16 16.04 0.5561
Ridge 395.16 16.04 0.5561
Lasso 395.16 16.04 0.5561
Random forest 219.24 10.54 0.7537 
Gradient boosted random forest 233.28 11.34 0.7379

## Murcko scaffold split
Regressor Test loss Test MAE Test R^2

MLP 440.67 16.22 0.4843
Linear 373.10 15.64 0.5634
Ridge 373.10 15.64 0.5634
Lasso 373.10 15.64 0.5634
Random forest 242.25 11.39 0.7165
Gradient boosted random forest 252.63 12.14 0.7044

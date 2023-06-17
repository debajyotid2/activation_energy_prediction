# Activation energy prediction using Multi-Layered Perceptrons

This repository deals with the task of predicting activation energy of reactions from fixed-width vector representations ("fingerprints") of the reactant and product molecules from their SMILES (Simplified Molecular Input Line-Entry System) representations. Simple regression models such as linear, ridge, LASSO and support vector regression are compared against more advanced models like random forests, gradient-boosted random forests and multi-layered perceptrons. 

Two datasets are used in this study:

1. Grambow et al (2020) (https://www.nature.com/articles/s41597-020-0460-4)
2. Reaction Graph Depth (RGD) Dataset (https://engineering.purdue.edu/savoiegroup/data+code.html)

### Dependencies

```
pytorch
pytorch-lightning
hydra
scikit-learn
xgboost
seaborn
matplotlib
numpy
rdkit
joblib
```
To run, first create the environment using either `pip` or `conda` (using `requirements.txt` or `environment.yml`, respectively, from the `environment` directory of this repo). Then, to run linear regression on the Grambow dataset with the default settings, run 
```
python run_sklearn.py
```
To run regression with the PyTorch MLP using the default settings, run
```
python run_pytorch.py
```

This project uses Hydra (https://github.com/facebookresearch/hydra) for configuration management. To learn more on how to use Hydra, please refer to documentation in Hydra's GitHub page.

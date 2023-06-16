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
seaborn
matplotlib
numpy
```

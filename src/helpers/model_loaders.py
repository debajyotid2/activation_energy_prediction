"""
Module containing functions to load the appropriate model 
as requested by the user.
"""

from typing import Any, Callable
import xgboost as xgb
from sklearn import linear_model, neural_network, svm, ensemble

MODELS = \
{
    "linear regression": linear_model.LinearRegression,
    "ridge regression": linear_model.Ridge,
    "lasso regression": linear_model.Lasso,
    "MLP": neural_network.MLPRegressor,
    "SVR": svm.SVR,
    "random forest": ensemble.RandomForestRegressor,
    "gradient boosted random forest": xgb.XGBRegressor,
}

def load_model(model_name: str) -> Callable[Any, Any]:
    """
    Loads a regressor model from a dictionary given model name.
    """
    if model_name not in MODELS:
        raise ValueError(f"{model_name} is not valid.")
    return MODELS[model_name]

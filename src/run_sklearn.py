"""
Workflow for running regression on activation energy prediction
using Scikit-learn models and Pandas datasets.
"""

import logging
from pathlib import Path
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from helpers.dataloaders import get_pandas_dataloader
from helpers.model_loaders import load_model
from helpers.sklearn_args import SklearnArgs

logging.basicConfig(
            format="%(asctime)s-%(levelname)s: %(message)s",
            level=logging.INFO)

cs = ConfigStore.instance()
cs.store(name="sklearn_args", node=SklearnArgs)

@hydra.main(config_path="../hydra_conf", 
            config_name="sklearn_config",
            version_base="1.3")
def main(cfg: SklearnArgs):
    # load data
    loader = get_pandas_dataloader(cfg.common_args.dataset, 
                                   cfg.common_args.split)
    if cfg.common_args.split == "random":
        X_train, Y_train, X_test, Y_test = loader(
                            Path(cfg.common_args.data_path),
                            cfg.dataloader_args.radius, 
                            cfg.dataloader_args.n_bits,
                            cfg.dataloader_args.test_frac, 
                            cfg.dataloader_args.seed,
                            cfg.common_args.data_url)
    else:
        X_train, Y_train, X_test, Y_test = loader(
                            Path(cfg.common_args.data_path),
                            cfg.dataloader_args.radius, 
                            cfg.dataloader_args.n_bits,
                            cfg.dataloader_args.val_frac,
                            cfg.dataloader_args.test_frac,
                            cfg.common_args.data_url)
 
    # load model
    model_obj = load_model(cfg.model_choice_args.model)
    if "random forest" in cfg.model_choice_args.model:
        model = model_obj(
                n_estimators=cfg.random_forest_reg_args.num_estimators)
    elif "MLP" in cfg.model_choice_args.model:
        model = model_obj(
                hidden_layer_sizes=(cfg.mlp_reg_args.layer_size, 
                                    cfg.mlp_reg_args.layer_size, ),
                learning_rate_init=cfg.mlp_reg_args.learning_rate,
                random_state=cfg.dataloader_args.seed,
                early_stopping=True,
                validation_fraction=cfg.dataloader_args.val_frac,
                max_iter=cfg.mlp_reg_args.max_iter,
                tol=cfg.mlp_reg_args.tol,
                n_iter_no_change=cfg.mlp_reg_args.n_iter_no_change,
                verbose=True
            )
    else:
        model = model_obj()

    logging.info("Starting training...")
    model.fit(X_train, Y_train)

    logging.info("Finished training.")
    logging.info(f"Test R-squared = {model.score(X_test, Y_test):.4f}")

    Y_pred = model.predict(X_test)
    mae_test = np.mean(np.abs(Y_pred-Y_test), axis=0)
    mse_test = np.mean((Y_pred-Y_test)**2, axis=0)

    logging.info(f"Test MAE = {mae_test:.4f} kcal/mol")
    logging.info(f"Test MSE = {mse_test:.4f} kcal/mol")

if __name__=="__main__":
    main()

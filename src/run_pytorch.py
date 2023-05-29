"""
Workflow for running regression on activation energy prediction
using PyTorch models and dataloaders.
"""

import logging
from pathlib import Path
import hydra
import torch
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from helpers.dataloaders import get_pytorch_dataloader
from helpers.pytorch_args import PyTorchArgs
from model.model import MLPRegressor

logging.basicConfig(
            format="%(asctime)s-%(levelname)s: %(message)s",
            level=logging.INFO)

cs = ConfigStore.instance()
cs.store(name="pytorch_args", node=PyTorchArgs)

@hydra.main(config_path="../hydra_conf", 
            config_name="pytorch_config",
            version_base="1.3")
def main(cfg: PyTorchArgs):
    torch.multiprocessing.set_sharing_strategy("file_system")

    # load data
    loader = get_pytorch_dataloader(cfg.common_args.dataset, 
                                   cfg.common_args.split)
    train_loader, val_loader, test_loader = loader(
                        data_dirpath=Path(cfg.common_args.data_path).parent,
                        data_url=cfg.common_args.data_url,
                        radius=cfg.dataloader_args.radius,
                        n_bits=cfg.dataloader_args.n_bits,
                        batch_size=cfg.dataloader_args.batch_size,
                        val_frac=cfg.dataloader_args.val_frac,
                        test_frac=cfg.dataloader_args.test_frac,
                        num_workers=cfg.dataloader_args.num_workers,
                        seed=cfg.dataloader_args.seed
                    )

    # load model
    model = MLPRegressor(num_input_nodes=cfg.dataloader_args.n_bits+1,
                         num_layers=cfg.mlpreg_args.num_layers,
                         num_nodes_per_layer=cfg.mlpreg_args.num_nodes_per_layer,
                         learning_rate=cfg.mlpreg_args.learning_rate,
                         loss_function=torch.nn.MSELoss())

    logging.info("Starting training...")

    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         logger=True, 
                         enable_progress_bar=True,
                         max_epochs=cfg.mlpreg_args.epochs)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    logging.info("Finished training.")
    
    trainer.test(model=model,
                 dataloaders=test_loader)

if __name__=="__main__":
    main()

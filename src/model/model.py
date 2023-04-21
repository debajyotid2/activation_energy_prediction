"""
PyTorch model for reaction property prediction.
"""
from typing import Any
import torch
import torchmetrics
import pytorch_lightning as pl

class MLPRegressor(pl.LightningModule):
    """
    Multi-layered perceptron (MLP) based model to predict reaction 
    properties like activation energy based on descriptors of reactants
    and products generated from SMILES.
    """
    def __init__(self,
                 num_input_nodes: int,
                 loss_function: Any,
                 num_layers: int = 2,
                 num_nodes_per_layer: int = 200,
                 learning_rate: float = 0.001,
                 dropout_prob: float=0.0):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_function"])
        
        top = [torch.nn.Linear(num_input_nodes, num_nodes_per_layer),
               torch.nn.ReLU()]
        middle = []
        for _ in range(num_layers-2):
            middle.append(*(torch.nn.Linear(num_nodes_per_layer, num_nodes_per_layer),
                        torch.nn.ReLU()))
        bottom = [torch.nn.Linear(num_nodes_per_layer, 1)]
        middle.extend(bottom)
        top.extend(middle)
        if dropout_prob > 0.0:
            top.extend([torch.nn.Dropout(p=dropout_prob)])
        self.model = torch.nn.Sequential(*top)

        self.loss = loss_function
        
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Model output.
        """
        return self.model(data)

    def training_step(self, 
                      batch: tuple[torch.Tensor, torch.Tensor], 
                      batch_idx: int| torch.Tensor) -> torch.Tensor:
        """
        Training step.
        """
        inputs, targets = batch
        preds = self.model(inputs)
        loss = self.loss(preds.view(-1), targets)
        self.train_r2(preds.view(-1), targets)
        self.train_mae(preds.view(-1), targets)
        self.log("train_loss", loss,
                 on_step=True, on_epoch=True)
        self.log("train_r2", self.train_r2)
        self.log("train_mae", self.train_mae)

        return loss

    def validation_step(self,
                        batch: tuple[torch.Tensor, torch.Tensor], 
                        batch_idx: int| torch.Tensor) -> torch.Tensor:
        """
        Validation step.
        """
        inputs, targets = batch
        preds = self.model(inputs)
        loss = self.loss(preds.view(-1), targets)
        self.val_r2(preds.view(-1), targets)
        self.val_mae(preds.view(-1), targets)
        self.log("val_loss", loss)
        self.log("val_r2", self.val_r2)
        self.log("val_mae", self.val_mae)

    def test_step(self,
                  batch: tuple[torch.Tensor, torch.Tensor], 
                  batch_idx: int| torch.Tensor) -> torch.Tensor:
        """
        Test step.
        """
        inputs, targets = batch
        preds = self(inputs)
        loss = self.loss(preds.view(-1), targets)
        self.test_r2(preds.view(-1), targets)
        self.test_mae(preds.view(-1), targets)

        self.log("test_loss", loss)
        self.log("test_r2", self.test_r2)
        self.log("test_mae", self.test_mae)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Initialize, configure and return optimizer.
        """
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.learning_rate)


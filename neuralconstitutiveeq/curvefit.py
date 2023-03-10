import pytorch_lightning as pl
import torch
from torch import nn, Tensor


class SimpleCurveFitter(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        # Manual optimization used to support second-order optimizers such as L-BFGS
        # self.automatic_optimization = False

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def compute_loss(self, batch) -> Tensor:
        x, y_true = batch
        y_pred = self(x.squeeze())
        loss = torch.nn.functional.mse_loss(y_true.squeeze(), y_pred)
        return loss

    def training_step(self, batch, batch_idx) -> Tensor:
        loss = self.compute_loss(batch)
        self.log_dict(dict(self.model.named_parameters()))
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.LBFGS(self.parameters(), lr=self.lr)

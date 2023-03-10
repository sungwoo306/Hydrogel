# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from neuralconstitutiveeq.tipgeometry import Conical
from neuralconstitutiveeq.forcemodel import PLRTriangularIndentation
from neuralconstitutiveeq.curvefit import SimpleCurveFitter

# %%
# scale 1um -> 1
# tip = Spherical(R=5e-8, h = 6.2e-6)
tip = Conical(np.pi / 10)
print(tip.alpha_corrections(6.2, 4))
print(tip.beta_corrections(4))

# %%
# scale 1um to 1
# multiply all length quantities by 1e6
# Pa = N/m^2 = kgm/s^2 /m^2 = kg/(ms^2) -> divide by 1e6

F_true = PLRTriangularIndentation(
    tip, 6.2, 4, v=10, t_max=0.2, E0=5.72e-1, t0=1, gamma=0.42
)
# %%
with torch.no_grad():
    t = torch.linspace(0.0, 0.4, 100)
    f_data = F_true(t)
dataset = torch.utils.data.TensorDataset(t.view(1, -1), f_data.view(1, -1))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

plt.plot(t.numpy(), f_data.clone().detach().numpy(), ".")

# %%
F_model = PLRTriangularIndentation(
    tip, 6.2, 4, v=10, t_max=0.2, E0=0.1, t0=1.0, gamma=0.5
)
F_model.t0.requires_grad = False
regression = SimpleCurveFitter(F_model, 0.1)

# %%
if __name__ == "__main__":
    pl.seed_everything(10)

    logger = WandbLogger(entity="jhelab", project="force_indentation")
    # logger = None
    checkpoint_every_n_epochs = 5
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(
                monitor="mse",
                mode="min",
                save_top_k=1,
                filename="best_{epoch}-{step}",
            ),
            ModelCheckpoint(
                monitor="mse",
                every_n_epochs=checkpoint_every_n_epochs,
                save_top_k=-1,
                filename="{epoch}-{step}",
            ),
        ],
        logger=logger,
        log_every_n_steps=1,
        auto_select_gpus=True,
        deterministic="warn",
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(regression, dataloader)

# %%
F_model.E0
# %%
dict(F_model.named_parameters())
# %%

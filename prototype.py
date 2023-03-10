# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults

from neuralconstitutiveeq.tipgeometry import Conical
from neuralconstitutiveeq.forcemodel import PLRTriangularIndentation
from neuralconstitutiveeq.curvefit import SimpleCurveFitter

configure_matplotlib_defaults()
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
# Resulting E0 (and thus F_true) was too small, so was then multiplied by 1e3
F_true = PLRTriangularIndentation(
    tip, 6.2, 4, v=10, t_max=0.2, E0=5.72e-1, t0=1, gamma=0.42
)
# %%
with torch.no_grad():
    t = torch.linspace(0.0, 0.4, 100)
    f_data = F_true(t)
dataset = torch.utils.data.TensorDataset(t.view(1, -1), f_data.view(1, -1))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t.numpy(), f_data.clone().detach().numpy(), ".")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Force (mN)")
ax.set_title(
    r"Simulated $F(t)$ for the PLR model with $E_0=0.572$ kPa, $\gamma=0.42$, and $t_0=1$ s",
    fontsize="medium",
)
# %%
F_model = PLRTriangularIndentation(
    tip, 6.2, 4, v=10, t_max=0.2, E0=0.1, t0=1.0, gamma=0.0
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
F_model.gamma
# %%
dict(F_model.named_parameters())
# %%
loaded_data = np.load("./Image00801.npz")
# %%
v = float(loaded_data["v"]) * 1e6
t_max = float(loaded_data["t_max"])
t_max
force = loaded_data["force"] * 1e9
time = loaded_data["time"]
force = force - force[0]
# %%
plt.plot(time, force)
# %%
dataset = torch.utils.data.TensorDataset(
    torch.tensor(time).view(1, -1), torch.tensor(force).view(1, -1)
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

# %%
tip = Conical(np.pi / 18)
F_model = PLRTriangularIndentation(
    tip, 1e3, 1, v=v, t_max=t_max, E0=10.0, t0=1.0, gamma=0.5
)
F_model.t0.requires_grad = False
regression = SimpleCurveFitter(F_model, 5e-3)
# %%
with torch.no_grad():
    f_init = F_model(torch.tensor(time))

plt.plot(time, f_init.clone().detach().numpy(), ".")
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
with torch.no_grad():
    f_fitted = F_model(torch.tensor(time)).clone().detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(time, force, label="Data")
axes[0].plot(time, f_fitted, label="Fitted")
axes[0].legend()
# %%

# %%
from configparser import ConfigParser

import numpy as np
from numpy import ndarray
import xarray as xr
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

# import kneed
from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults

configure_matplotlib_defaults()

filepath = (
    "./data/afm/20230106/highly entangled hydrogel(C=1.0)/Image00801.nid"
)
config, data = nanosurf.read_nid(filepath)


# %%
def get_sampling_rate(nid_config: ConfigParser) -> float:
    spec_config = dict(config[r"DataSet\DataSetInfos\Spec"])
    num_points = int(spec_config["data points"])
    # May later use the pint library to parse unitful quantites
    modulation_time = float(spec_config["modulation time"].split(" ")[0])
    return num_points / modulation_time


get_sampling_rate(config)
# %%
forward, backward = data["spec forward"], data["spec backward"]

forward


# %%
def get_z_and_defl(spectroscopy_data: xr.DataArray) -> tuple[ndarray, ndarray]:
    piezo_z = spectroscopy_data["z-axis sensor"].to_numpy()
    defl = spectroscopy_data["deflection"].to_numpy()
    return piezo_z.squeeze(), defl.squeeze()


def calc_tip_distance(piezo_z_pos: ndarray, deflection: ndarray) -> ndarray:
    return piezo_z_pos - deflection


def find_contact_point(distance: ndarray, deflection: ndarray) -> float:
    # Right now, only support 1D arrays of tip_distance and tip_deflection
    locator = kneed.KneeLocator(
        distance,
        deflection,
        S=1,
        curve="convex",
        direction="increasing",
        interp_method="polynomial",
        polynomial_degree=7,
    )
    return locator.knee


def fit_baseline_polynomial(
    distance: ndarray,
    deflection: ndarray,
    contact_point: float = 0.0,
    degree: int = 1,
) -> Polynomial:
    pre_contact = distance < contact_point
    domain = (np.amin(distance), np.amax(distance))
    return Polynomial.fit(
        distance[pre_contact],
        deflection[pre_contact],
        deg=degree,
        domain=domain,
    )


# %%
z_fwd, defl_fwd = get_z_and_defl(forward)
z_bwd, defl_bwd = get_z_and_defl(backward)
dist_fwd = calc_tip_distance(z_fwd, defl_fwd)
dist_bwd = calc_tip_distance(z_bwd, defl_bwd)
# cp = find_contact_point(dist_fwd, defl_fwd)
#%%
def distance_std(N1: int):
    N2 = N1 * 2
    std_bwd_front = np.std(dist_bwd[:N1])
    std_bwd_rear = np.std(dist_bwd[N1:N2])
    return np.abs(std_bwd_front - std_bwd_rear)


N_f = 100
N_b = 100
d_array_f = []
d_array_b = []
for i, j in zip(range(N_f), range(N_b)):
    d_f = distance_std(i)
    d_b = distance_std(j)
    d_array_f.append(d_f)
    d_array_b.append(d_b)

max_diff_f = np.argmax(d_array_f[1:])
max_diff_b = np.argmax(d_array_b[1:])
max_diff_f, max_diff_b
#%%
n_f = max_diff_f * 2
n_b = max_diff_b * 2
A_f = np.stack((dist_fwd[:n_f], np.ones(len(dist_fwd[:n_f]))), axis=0).T
m_f, c_f = np.linalg.lstsq(A_f, defl_fwd[:n_f], rcond=None)[0]

A_b = np.stack((dist_bwd[:n_b], np.ones(len(dist_bwd[:n_b]))), axis=0).T
m_b, c_b = np.linalg.lstsq(A_b, defl_bwd[:n_b], rcond=None)[0]
m_f, c_f, m_b, c_b
#%%

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(dist_fwd, defl_fwd, ".", label="forward", markersize=3)
ax.plot(dist_bwd, defl_bwd, ".", label="backward", markersize=3)
ax.plot(dist_bwd, m_f * dist_bwd + c_f, ".", label="linear_reg", markersize=3)
ax.plot(
    dist_fwd,
    defl_fwd - (m_f * dist_fwd + c_f),
    ".",
    label="base_line",
    markersize=3,
)
ax.plot(dist_bwd, m_b * dist_bwd + c_b, ".", label="linear_reg", markersize=3)
ax.plot(
    dist_bwd,
    defl_bwd - (m_b * dist_bwd + c_b),
    ".",
    label="base_line",
    markersize=3,
)
ax.legend()
#%%
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(
    dist_fwd,
    defl_fwd - (m_f * dist_fwd + c_f),
    ".",
    label="base_line_fwd",
    markersize=3,
)
ax.plot(
    dist_bwd,
    defl_bwd - (m_b * dist_bwd + c_b),
    ".",
    label="base_line_bwd",
    markersize=3,
)
ax.legend()
#%%
dist_total = np.concatenate((dist_fwd, dist_bwd[::-1]), axis=-1)
defl_total = np.concatenate((defl_fwd, defl_bwd[::-1]), axis=-1)
is_contact = dist_total >= 0
indentation = dist_total[is_contact]
k = 0.2  # N/m
force = defl_total[is_contact] * k
sampling_rate = get_sampling_rate(config)
time = np.arange(len(indentation)) / sampling_rate

print(indentation.shape)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(time, indentation)
axes[1].plot(time, force)
axes[1].set_xlabel("$Time$ $t$")

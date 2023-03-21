# %%
from configparser import ConfigParser

import numpy as np
from numpy import ndarray
import xarray as xr
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
import kneed
from jhelabtoolkit.io.nanosurf import nanosurf
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults

configure_matplotlib_defaults()

filepath = "./data/afm/20230106/highly entangled hydrogel(C=1.0)/Image00801.nid"
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


def find_contact_point1(distance: ndarray, deflection: ndarray) -> float:
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

def find_contact_point2(deflection: ndarray, N: int) -> ndarray:
    # Ratio of Variance
    rov = np.array([])
    length = np.arange(np.size(deflection))
    rov = np.array([np.append(rov, np.array([np.var(deflection[i+1:i+N])/np.var(deflection[i-N:i-1])])) for i in length]).flatten()
    rov = rov[N:np.size(rov)-N]
    idx = np.argmax(rov)
    return rov, idx, rov[idx]

def fit_baseline_polynomial(
    distance: ndarray, deflection: ndarray, contact_point: float = 0.0, degree: int = 1
) -> Polynomial:
    pre_contact = distance < contact_point
    domain = (np.amin(distance), np.amax(distance))
    return Polynomial.fit(
        distance[pre_contact], deflection[pre_contact], deg=degree, domain=domain
    )
# %%
z_fwd, defl_fwd = get_z_and_defl(forward) 
z_bwd, defl_bwd = get_z_and_defl(backward) 
dist_fwd = calc_tip_distance(z_fwd, defl_fwd) 
dist_bwd = calc_tip_distance(z_bwd, defl_bwd) 
# cp = find_contact_point(dist_fwd, defl_fwd)
#%%
# ROV method
N = 10
rov_fwd = find_contact_point2(defl_fwd, N)[0]
idx_fwd = find_contact_point2(defl_fwd, N)[1]
rov_fwd_max = find_contact_point2(defl_fwd, N)[2]

rov_bwd = find_contact_point2(defl_bwd, N)[0]
idx_bwd = find_contact_point2(defl_bwd, N)[1]
rov_bwd_max = find_contact_point2(defl_bwd, N)[2]
#%%
fig, ax = plt.subplots(1, 1, figsize = (7, 5))
ax.plot(dist_fwd[N:np.size(dist_fwd)-N], find_contact_point2(defl_fwd, N)[0])
ax.set_xlabel("Distance(forward)")
ax.set_ylabel("ROV")
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd, defl_fwd, label="forward")
ax.plot(dist_bwd, defl_bwd, label="backward")
ax.legend()
# %%
# Find contact point
cp_fwd = dist_fwd[N+idx_fwd]
cp_bwd = dist_bwd[N+idx_bwd]
print(cp_fwd, cp_bwd)
#%%
# Translation
dist_fwd = dist_fwd - cp_fwd
dist_bwd = dist_bwd - cp_fwd

#%%
# Polynomial fitting
baseline_poly_fwd = fit_baseline_polynomial(dist_fwd, defl_fwd)
defl_processed_fwd = defl_fwd - baseline_poly_fwd(dist_fwd)
baseline_poly_bwd = fit_baseline_polynomial(dist_bwd, defl_bwd)
defl_processed_bwd = defl_bwd - baseline_poly_bwd(dist_bwd)

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd, defl_fwd, label="forward")
ax.plot(dist_bwd, defl_bwd, label="backward")
plt.axvline(cp_fwd, color='grey', linestyle='--', linewidth=1)
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd, defl_processed_fwd, label="forward")
ax.plot(dist_bwd, defl_processed_bwd, label="backward")
plt.axvline(cp_fwd, color='grey', linestyle='--', linewidth=1)
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(dist_fwd, z_fwd, label="forward")
ax.plot(dist_bwd, z_bwd, label="backward")
ax.legend()
# %%
dist_total = np.concatenate((dist_fwd, dist_bwd[::-1]), axis=-1)
defl_total = np.concatenate((defl_fwd, defl_bwd[::-1]), axis=-1)
is_contact = dist_total >= 0
indentation = dist_total[is_contact]
k = 0.2  # N/m
force = defl_total[is_contact] * k
sampling_rate = get_sampling_rate(config)
time = np.arange(len(indentation)) / sampling_rate

fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
axes[0].plot(time, indentation)
axes[0].set_xlabel("Time(s)")
axes[1].set_xlabel("Time(s)")
axes[0].set_ylabel("Indentation(m)")
axes[1].set_ylabel("Force(N)")
axes[1].plot(time, force)
# %%
max_ind = np.argmax(indentation)
t_max = time[max_ind]
t_max
indent_max = indentation[max_ind]
# %%
Polynomial.fit(
    time[: max_ind + 1],
    indentation[: max_ind + 1],
    deg=[
        1,
    ],
).convert()
# %%
Polynomial.fit(
    time[max_ind + 1 :],
    indentation[max_ind + 1 :],
    deg=[
        1,
    ],
).convert()
# %%
indent_max / t_max
# %%
indent_max / (time[-1] - t_max)
# %%
v_avg = 2 * indent_max / time[-1]
# %%
np.savez(
    "./Image00801.npz",
    indentation=indentation,
    time=time,
    force=force,
    v=v_avg,
    t_max=t_max,
)

# %%
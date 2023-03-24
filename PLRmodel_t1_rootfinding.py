#%%
from functools import partial
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults
configure_matplotlib_defaults()
#%%
# PLR analytic solution of t1
def t1_analytic(t, t_max, gamma):
    return np.clip(t - 2**(1/(1-gamma))*(t-t_max), 0, None)
# %%
def PLR_Relaxation_function(t_prime, t, v, t0, E0, gamma) -> float:
    return v * E0 * (((t-t_prime)/t0)**(-gamma))

#%%
def PLR_Integration(t1, t, v, t_max, t0, E0, gamma) -> float:
    res = quad(PLR_Relaxation_function, t1, t_max, args=(t, v, t0, E0, gamma))[0] - quad(PLR_Relaxation_function, t_max, t, args=(t, v, t0, E0, gamma))[0]    
    return res

#%%
def KWW_Relaxation_function(t_prime, t, tau, beta) :
    return np.exp(-((t-t_prime)/(tau))**beta)

#%%
def KWW_Integration(t1, t, t_max, tau, beta):
    res = quad(KWW_Integration, t1, t_max, args=(t, tau, beta))[0] - quad(KWW_Integration, t_max, t, args=(t, tau, beta))[0]
    return res
#%%
# PLR model
integration = partial(PLR_Integration, t0 = 1, v = 10, E0 = 572, t_max = 0.2, gamma = 0.42)

#%%
print(integration(0.17696173488041395, 0.21))
print(t1_analytic(0.21, 0.2, 0.42))
#%%
t_final = 0.4
t_array = np.linspace(0.21, t_final, 100)
t1_array = np.array([root_scalar(integration, x0=0.0, x1=0.2, args=(t), method='secant').root for t in t_array])
t1_array = np.clip(t1_array, 0, None)

t1_analytic_array = t1_analytic(t_array, 0.2, 0.42)

# t1_array = np.array([root_scalar(integration(t), 0, method='newton')] for t in t_array)
# print(len(t1_array))
fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.plot(t_array, t1_array, 'o', label="expect", markerfacecolor = "white", alpha = 0.8)
ax.plot(t_array, t1_analytic_array, label="analytic", linewidth = 2, alpha = 0.8)
ax.legend()
# %%
# KWW model
integration = partial(KWW_Integration, t_max = 0.2, tau = 0.1, beta = 8)

#%%
integration(1,2)
#%%
t_final = 0.4
t_array = np.linspace(0.21, t_final, 100)
t1_array = np.array([root_scalar(integration, x0=0.0, x1=0.2, args=(t), method='secant').root for t in t_array])
t1_array = np.clip(t1_array, 0, None)

t1_analytic_array = t1_analytic(t_array, 0.2, 0.42)
#%%
# t1_array = np.array([root_scalar(integration(t), 0, method='newton')] for t in t_array)
# print(len(t1_array))
fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.plot(t_array, t1_array, 'o', label="expect", markerfacecolor = "white", alpha = 0.8)
ax.plot(t_array, t1_analytic_array, label="analytic", linewidth = 2, alpha = 0.8)
ax.legend()

# %%

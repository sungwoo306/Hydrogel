# KWW model
#%%
from functools import partial
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults
configure_matplotlib_defaults()
#%%
def KWW_analytic(t, t_max, beta):
    return 
#%%
def KWW_Relaxation_function(t_prime, t, tau, beta) :
    return np.exp(-((t-t_prime)/(tau))**beta)

#%%
def KWW_Integration(t1, t, t_max, tau, beta):
    res = quad(KWW_Relaxation_function, t1, t_max, args=(t, tau, beta))[0] - quad(KWW_Relaxation_function, t_max, t, args=(t, tau, beta))[0]
    return res
#%%
integration = partial(KWW_Integration, t_max = 0.2, tau = 1, beta = 5.0)
#%%
t_final = 0.4
t_array = np.linspace(0.21, t_final, 100)
t1_array = np.array([root_scalar(integration, x0=0.0, x1=0.2, args=(t), method='secant').root for t in t_array])
t1_array = np.clip(t1_array, 0, 1e+5)
print(t1_array)

# t1_analytic_array = t1_analytic(t_array, 0.2, 0.42)
#%%
# t1_array = np.array([root_scalar(integration(t), 0, method='newton')] for t in t_array)
# print(len(t1_array))
fig, ax = plt.subplots(1, 1, figsize=(10,7))
ax.plot(t_array, t1_array, 'o', label="expect", markerfacecolor = "white", alpha = 0.8)
# ax.plot(t_array, t1_analytic_array, label="analytic", linewidth = 2, alpha = 0.8)
ax.legend()

# %%

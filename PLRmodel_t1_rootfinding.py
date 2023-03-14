#%%
from functools import partial
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
#%%
# PLR analytic solution of t1
def t1_analytic(t, t_max, gamma):
    return np.clip(t - 2**(1/(1-gamma))*(t-t_max), 0, None)
# %%
def Integrand(t_prime, t, v, t0, E0, gamma) -> float:
    return v * E0 * (((t-t_prime)/t0)**(-gamma))

#%%
def PLR_Integration(t1, t, v, t_max, t0, E0, gamma) -> float:
    res = quad(Integrand, t1, t_max, args=(t, v, t0, E0, gamma))[0] - quad(Integrand, t_max, t, args=(t, v, t0, E0, gamma))[0]    
    return res
#%%
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
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(t_array, t1_array, 'o', label="expect", markerfacecolor = "white", alpha = 0.8)
ax.plot(t_array, t1_analytic_array, label="analytic", linewidth = 2, alpha = 0.8)
ax.legend()
# %%

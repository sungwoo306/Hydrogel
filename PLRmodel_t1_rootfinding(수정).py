#%%
from functools import partial
import numpy as np
import torch
from torch import nn
from xitorch.integrate import quad
from xitorch.optimize import rootfinder
import matplotlib.pyplot as plt
from jhelabtoolkit.utils.plotting import configure_matplotlib_defaults
configure_matplotlib_defaults()
#%%
# PLR analytic solution of t1
def t1_analytic(t, t_max, gamma):
    return torch.clip(t - 2**(1/(1-gamma))*(t-t_max), 0, None)
# %%
class Integrand(nn.Module):
    def __init__(self, t, t0, v, E0, gamma):
        super().__init__()
        self.t = t
        self.t0 = t0
        self.v = v
        self.E0 = E0
        self.gamma = gamma
        
    def forward(self, t_prime):
        Integrand = self.v * self.E0 * (((self.t-t_prime)/self.t0)**(-self.gamma))
        return Integrand
    
# def PLR_Integration(t1, t, v, t_max, t0, E0, gamma) -> float:
#     res = quad(Integrand, t1, t_max, args=(t, v, t0, E0, gamma)) - quad(Integrand, t_max, t, args=(t, v, t0, E0, gamma))    
#     return res
#%%
# t1 = torch.Tensor([0.17696173488041395])
# t = torch.Tensor([0.21])
# t_final = 0.4
# t_array = torch.linspace(0.21, t_final, 100)
# # t1_array = torch.Tensor([rootfinder(integration, x0=0.0, x1=0.2, args=(t), method='secant') for t in t_array])
# t1_array = torch.clip(t1_array, 0, None)
# t1_analytic_array = t1_analytic(t_array, 0.2, 0.42)

t0 = torch.Tensor([1])
t_max = torch.Tensor([0.2])
v = torch.Tensor([10])
E0 = torch.Tensor([572])
gamma = torch.Tensor([0.42])
t_max = torch.Tensor([0.2])
t = torch.Tensor([0.2])
y = torch.zeros([1])
t1 = torch.Tensor([0.5])
a = quad(Integrand(t=t, t0 = t0, v = v, E0 = E0, gamma = gamma), t1, t_max) - quad(Integrand(t = t, t0 = t0, v = v, E0 = E0, gamma = gamma), t_max, t)
t1 = torch.Tensor(rootfinder(quad((Integrand(t=t, t0 = t0, v = v, E0 = E0, gamma = gamma), t1, t_max) - quad(Integrand(t = t, t0 = t0, v = v, E0 = E0, gamma = gamma), t_max, t)), y0 = y, params=(t, v, t0, E0, gamma)))

print(a)
#%%
def integrand(x, a, b):
    return torch.Tensor([a*x**2 + b])
lb = torch.Tensor([0])
lu = torch.Tensor([1])
a = torch.Tensor([[2.0]]).requires_grad_()
b = torch.Tensor([[1.0]]).requires_grad_()
y = torch.zeros([1])
print(quad(integrand, lb, lu, (a,b)))

root = rootfinder(integrand, y0=y, params=(a,b))
print(root)

def func1(y, A):  # example function
    return torch.tanh(A @ y + 0.1) + y / 2.0
A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
y0 = torch.zeros((2,1))  # zeros as the initial guess
yroot = rootfinder(func1, y0, params=(A,))
print(yroot)
# print(root)
#%%

#%%
# integration= Integrand(0.17696173488041395, 0.21, t0, v, E0, gamma)
a = torch.Tensor([0.17696173488041395])
b = torch.Tensor([0.21])
print(integration(a, b))
#%%
# print(t1_analytic(0.21, 0.2, 0.42))
#%%
t_final = torch.Tensor([0.4])
t_array = torch.arange(0.21, 0.4, 100)
t1_array = [rootfinder(integration, y0=0.0, params=(t), method='broyden1') for t in t_array]
t1_array = torch.clip(t1_array, 0, None)

t1_analytic_array = t1_analytic(t_array, 0.2, 0.42)
#%%
# t1_array = np.array([root_scalar(integration(t), 0, method='newton')] for t in t_array)
# print(len(t1_array))
fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(t_array, t1_array, 'o', label="expect", markerfacecolor = "white", alpha = 0.8)
ax.plot(t_array, t1_analytic_array, label="analytic", linewidth = 2, alpha = 0.8)
ax.legend()
# %%

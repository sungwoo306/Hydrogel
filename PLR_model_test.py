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
    
#%%
def PLR_Integration(t1, t, v, t_max, t0, E0, gamma) -> torch.Tensor:
    res = quad(Integrand, t1, t_max, params=(t1, t, v, t0, E0, gamma)) - quad(Integrand, t_max, t, params=(t1, t, v, t0, E0, gamma))    
    return res
#%%
class PLRmodel(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t0 = torch.nn.Parameter(torch.Tensor([1]))
        self.t_max = torch.nn.Parameter(torch.Tensor)([0.2])
        self.v = torch.nn.Parameter(torch.Tensor([10]))
        self.E0 = torch.nn.Parameter(torch.Tensor([572]))
        self.gamma = torch.nn.Parameter(torch.Tensor([0.42]))
        self.t = t

    def forward(self, t_prime):
        Integrand = self.v * self.E0 * (((self.t-t_prime)/self.t0)**(-self.gamma))
        PLR_Integration











#%%
# Using rootfinder
def integrand(x, a, b):
    return torch.Tensor([a*x**2 + b])
lb = torch.Tensor([0])
lu = torch.Tensor([1])
a = torch.Tensor([[2.0]]).requires_grad_()
b = torch.Tensor([[-1.0]]).requires_grad_()
y = torch.zeros([1])
print(quad(integrand, lb, lu, (a,b)))

root = rootfinder(integrand, y0=y, params=(a,b))
print(root)
#%%

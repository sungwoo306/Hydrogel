import torch
from torch import nn

from neuralconstitutiveeq.tipgeometry import TipGeometry
from neuralconstitutiveeq.utils import beta, to_parameter

"""
Currently the forward method only works for 0D and 1D tensors
Will need to extend so that this works for any number of batch dimensions
"""


class PLRTriangularIndentation(nn.Module):
    def __init__(
        self,
        tip: TipGeometry,
        h: float,
        approx_order: int,
        v: float,
        t_max: float,
        E0: float,
        gamma: float,
        t0: float = 1.0,
    ):
        super().__init__()
        self.tip = tip
        alphas = self.tip.alpha_corrections(h, approx_order)
        betas = self.tip.beta_corrections(approx_order)
        self.register_buffer("alphas", torch.tensor(alphas))
        self.register_buffer("betas", torch.tensor(betas))
        self.v = v
        self.t_max = t_max
        self.E0 = to_parameter(E0)
        self.t0 = to_parameter(t0)
        self.gamma = to_parameter(gamma)

    def t1(self, t):
        t1 = t - (t - self.t_max) * torch.pow(2.0, (1 - self.gamma).reciprocal())
        return torch.relu(t1)

    def _approach(self, t):
        t = t.unsqueeze(-1)
        coeff = self.alphas * self.betas * beta(self.betas, 1 - self.gamma)
        term1 = (self.v) ** self.betas * (self.t0) ** self.gamma
        term2 = t ** (self.betas - self.gamma)
        return self.E0 * torch.sum(coeff * term1 * term2, dim=-1)

    def forward(self, t):
        t = t.view(-1)
        is_app = t <= self.t_max
        t_app, t_ret = t[is_app], t[~is_app]
        F_app = self._approach(t_app)
        F_ret = self._approach(self.t1(t_ret))
        return torch.cat([F_app, F_ret], dim=-1)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class PCENTransform(nn.Module):
    def __init__(
        self, 
        eps=1E-6, 
        s=0.025, 
        alpha=0.98, 
        delta=2, 
        r=0.25, 
    ):
        super().__init__()
        # smoothing parameter
        self.s = torch.nn.Parameter(Tensor([s]))
        torch.nn.init.constant(self.s, s)

        # AGC strength
        self.alpha = torch.nn.Parameter(Tensor([alpha]))
        torch.nn.init.constant(self.alpha, alpha)

        # stabilised root compression using delta and r
        self.delta = torch.nn.Parameter(Tensor([delta]))
        torch.nn.init.constant(self.delta, delta)

        self.r = torch.nn.Parameter(Tensor([r]))
        torch.nn.init.constant(self.r, r)

        # arbitrary constant to avoid division by 0
        self.eps = eps

    def iir(self, x) -> Tensor:
        s = torch.clamp(self.s, min=0.0, max=1.0)
        M = [x[..., 0]]
        for t in range(1, x.size(-1)):
            m = (1. - s) * M[t - 1] + s * x[..., t]
            M.append(m)
        M = torch.stack(M, dim=-1)
        return M

    def forward(self,
                E: torch.Tensor,
                ):
        """
        :param xs:      Input tensor (#batch, time, idim).
        :param xs_mask: Input mask (#batch, 1, time).
        :return:
        """
        alpha = torch.min(self.alpha, torch.ones(
            self.alpha.size(), device=self.alpha.device))
        r = torch.max(self.r, torch.ones(
            self.r.size(), device=self.r.device))

        M = self.iir(E)
        #pcen = xs.div_(smoother.add_(self.eps).pow_(self.alpha)).add_(self.delta).pow_(self.r).sub_(self.delta**self.r)
        #pcen = ((xs / ((self.eps + smoother)**self.alpha) + self.delta)**(1./self.r)
        #      - self.delta**(1./self.r))
        pcen = (((E / ((self.eps + M)**self.alpha)) + self.delta)**self.r) - (self.delta**self.r)
        pcen = 2 * (pcen - pcen.min()) / (pcen.max() - pcen.min()) - 1

        return pcen
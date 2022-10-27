import datetime
from typing import Tuple

import numpy as np
import torch
from d3rlpy.models.torch.distributions import Distribution


def p_print(data):
    now = datetime.datetime.now()
    if now.microsecond % 12 == 0:
        print(data)
    else:
        pass


def gumbel_pdf(x, loc: float, scale) -> torch.Tensor:
    """Returns Gumbel's PDF with parameters loc and scale at x."""
    # substitute
    z = (x - loc) / scale

    return (1. / scale) * (torch.exp(-(z + (torch.exp(-z)))))


def gumbel_cdf(x, loc, scale) -> torch.Tensor:
    """Returns the value of Gumbel's cdf with parameters loc and scale at x."""
    z = (x - loc) / scale
    return torch.exp(-torch.exp(-z))


def trunc_GBL(p, x):
    threshold = p[0]
    loc = p[1]
    scale = p[2]
    x1 = x[x < threshold]
    nx2 = len(x[x >= threshold])
    L1 = (-torch.log((gumbel_pdf(x1, loc, scale) / scale))).sum()
    L2 = (-torch.log(1 - gumbel_cdf(threshold, loc, scale))) * nx2
    # print x1, nx2, L1, L2
    return L1 + L2


class GumbelDistribution(Distribution):
    def __init__(self, logits, probs=None, temperature=1):
        super().__init__()
        self.logits = logits
        self.probs = probs
        self.eps = 1e-20
        self.temperature = 1

    def sample_gumbel(self):
        U = torch.zeros_like(self.logits)
        U.uniform_(0, 1)
        to_gumbel = -torch.log(-torch.log(U + self.eps) + self.eps)
        return to_gumbel

    def gumbel_softmax_sample(self, logits = None):
        """
        Draw a sample from the Gumbel-Softmax distribution. The returned sample will be
        a probability distribution that sums to 1 across classes.
        """
        y = self.logits + self.sample_gumbel()
        out = torch.softmax(y / self.temperature, dim=-1)
        return out

    def sample_with_log_prob(self) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.rsample()
        return y, self.log_prob()

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (torch.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample().clamp(-1,1)

    def sample(self):
        return self.rsample().detach()

    def sample_n(self, n: int) -> torch.Tensor:
        samples = torch.from_numpy(np.array([self.rsample() for _ in range(n)]).reshape(n,))
        return samples

    def sample_n_with_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sample_n(n)
        return x, self.log_prob(x)

    def hard_sample(self):
        out = self.hard_gumbel_softmax_sample()
        return out

    def log_prob(self) -> torch.Tensor:
        y = self.sample()
        return torch.log(y + self.eps)

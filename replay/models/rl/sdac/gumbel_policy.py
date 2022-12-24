from typing import Tuple, Union, cast

import torch
from d3rlpy.models.torch import Encoder, CategoricalPolicy
from torch import nn

from replay.models.rl.sdac.gumbel_distribution import GumbelDistribution


class GumbelPolicy(CategoricalPolicy):

    _encoder: Encoder
    _action_size: int
    _min_logstd: float
    _max_logstd: float
    _use_std_parameter: bool
    _mu: nn.Linear
    _logstd: Union[nn.Linear, nn.Parameter]

    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__(encoder, action_size)
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        dist = self.dist(x)
        if deterministic:
            picked_actions = dist.sample()
            picked_actions_hard = (torch.max(picked_actions, dim=-1, keepdim=True)[0] == picked_actions).float()
            picked_actions = (picked_actions_hard - picked_actions).detach() + picked_actions
            picked_actions = picked_actions.argmax(axis = 1)
        else:
            picked_actions, log_prob = dist.sample_with_log_prob()
            picked_actions_hard = (torch.max(picked_actions, dim=-1, keepdim=True)[0] == picked_actions).float()
            picked_actions = (picked_actions_hard - picked_actions).detach() + picked_actions

        #    print(picked_actions)
        return (picked_actions, log_prob) if with_log_prob else picked_actions

    def dist(self, x: torch.Tensor) -> GumbelDistribution:
        h = self._encoder(x)
        h = self._fc(h)
        h = self.softmax(h)
        return GumbelDistribution(h)

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x, deterministic=True)
        # out = torch.argmax(out, dim=1)
        return cast(torch.Tensor, out)

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        return dist.log_prob()

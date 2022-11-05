from typing import Sequence, Optional

import torch
from d3rlpy.algos.torch import SACImpl
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.torch_utility import TorchMiniBatch

from replay.models.rl.sdac.gumbel_policy import GumbelPolicy


class SDACImpl(SACImpl):
    _policy: Optional[GumbelPolicy]
    _targ_policy: Optional[GumbelPolicy]

    def _build_actor(self) -> None:
        self._policy = create_gumbel_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None

        with torch.no_grad():
            action, log_prob = self._policy.sample_with_log_prob(
                batch.next_observations
            )
            entropy = self._log_temp().exp() * log_prob
            target = self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return (target - entropy).mean(dim=1).reshape(-1, 1)


def create_gumbel_policy(
        observation_shape: Sequence[int],
        action_size: int,
        encoder_factory: EncoderFactory,
) -> GumbelPolicy:
    encoder = encoder_factory.create(observation_shape)
    return GumbelPolicy(encoder, action_size)

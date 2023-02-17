from typing import Sequence, Optional

import torch
from d3rlpy.algos.torch import SACImpl
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.torch import (
    EnsembleContinuousQFunction, ContinuousQFunction, VectorEncoderWithAction
)
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

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        # apply fix to the encoder marking it discrete for actions when computing loss,
        # because that's what SDAC has
        # TODO: investigate, why we doesn't need it during compute_target
        self._switch_q_funcs_encoder_discreteness(True)
        result = super().compute_critic_loss(batch, q_tpn)
        self._switch_q_funcs_encoder_discreteness(False)
        return result

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

    def _switch_q_funcs_encoder_discreteness(self, new_value: bool):
        if self._q_func is not None and isinstance(self._q_func, EnsembleContinuousQFunction):
            for q_func in self._q_func.q_funcs:
                if (
                        isinstance(q_func, ContinuousQFunction)
                        and hasattr(q_func, '_encoder')
                        and isinstance(q_func._encoder, VectorEncoderWithAction)
                ):
                    q_func._encoder._discrete_action = new_value


def create_gumbel_policy(
        observation_shape: Sequence[int],
        action_size: int,
        encoder_factory: EncoderFactory,
) -> GumbelPolicy:
    encoder = encoder_factory.create(observation_shape)
    return GumbelPolicy(encoder, action_size)

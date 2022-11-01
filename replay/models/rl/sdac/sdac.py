from typing import Any, Optional, Sequence

import d3rlpy
import torch
import torch.nn.functional as F
from d3rlpy.argument_utility import (
    EncoderArg, QFuncArg, UseGPUArg, ScalerArg, ActionScalerArg,
    RewardScalerArg
)
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from d3rlpy.models.torch.encoders import _VectorEncoder, EncoderWithAction, VectorEncoder
from pyspark.sql import DataFrame
from torch import nn

import replay.models.rl.sdac.sdac_impl as sdac_impl
from replay.models.rl.rl_recommender import RLRecommender


class VectorEncoderWithAction(_VectorEncoder, EncoderWithAction):

    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        concat_shape = (observation_shape[0] + action_size,)
        super().__init__(
            observation_shape=concat_shape,
            hidden_units=hidden_units,
            use_batch_norm=use_batch_norm,
            use_dense=use_dense,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self._observation_shape = observation_shape

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self.action_size
            ).float()
        #raise Exception(x.shape, action.shape)
        try:
            x = torch.cat([x, action], dim=1)
        except:
            one_hot = F.one_hot(action.to(torch.int64).view(-1), num_classes=self.action_size)
            x = torch.cat([x, one_hot], dim=1)
            #raise Exception(one_hot.shape, one_hot)
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    @property
    def action_size(self) -> int:
        return self._action_size


class CustomEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom"

    def __init__(self, feature_size):
        self.feature_size = feature_size

    def create(self, observation_shape):
        return VectorEncoder(observation_shape, [self.feature_size, self.feature_size])

    def create_with_action(self, observation_shape, action_size):
        return VectorEncoderWithAction(
            observation_shape, action_size, [self.feature_size, self.feature_size]
        )

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}


class SDAC(RLRecommender):
    r"""FIXME: add docstring"""

    model: sdac_impl.SDAC

    _search_space = {
        "actor_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "critic_learning_rate": {"type": "loguniform", "args": [3e-5, 3e-4]},
        "n_epochs": {"type": "int", "args": [3, 20]},
        "temp_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "gamma": {"type": "loguniform", "args": [0.9, 0.999]},
        "n_critics": {"type": "int", "args": [2, 4]},
    }

    def __init__(
            self, *,
            top_k: int, n_epochs: int = 1,
            action_randomization_scale: float = 0.,
            use_negative_events: bool = False,
            rating_based_reward: bool = False,
            rating_actions: bool = False,
            reward_top_k: bool = False,
            test_log: DataFrame = None, 

            # SDAC inner params
            actor_learning_rate: float = 3e-4,
            critic_learning_rate: float = 3e-4,
            temp_learning_rate: float = 3e-4,
            actor_optim_factory: OptimizerFactory = AdamFactory(),
            critic_optim_factory: OptimizerFactory = AdamFactory(),
            temp_optim_factory: OptimizerFactory = AdamFactory(),
            q_func_factory: QFuncArg = "mean",
            batch_size: int = 256,
            n_frames: int = 1,
            n_steps: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            n_critics: int = 2,
            initial_temperature: float = 1.0,
            use_gpu: UseGPUArg = False,
            scaler: ScalerArg = None,
            action_scaler: ActionScalerArg = None,
            reward_scaler: RewardScalerArg = None,
            actor_encoder_factory: EncoderArg = "default",
            critic_encoder_factory: EncoderArg = "default",
            **params: Any
    ):
        model = sdac_impl.SDAC(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factory=CustomEncoderFactory(256), 
            critic_encoder_factory=CustomEncoderFactory(256),
            encoder_factory=CustomEncoderFactory(256),
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            impl=None,
            **params
        )

        super(SDAC, self).__init__(
            model=model,
            test_log=test_log,
            top_k=top_k, n_epochs=n_epochs,
            action_randomization_scale=0,
            use_negative_events=use_negative_events,
            rating_based_reward=rating_based_reward,
            rating_actions=rating_actions,
            reward_top_k=reward_top_k,
        )

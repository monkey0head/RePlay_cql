from typing import Sequence, Optional, Any

import torch
from d3rlpy.algos import SAC
from d3rlpy.gpu import Device
from d3rlpy.algos.torch import SACImpl
from d3rlpy.argument_utility import (
    EncoderArg, QFuncArg, UseGPUArg, ScalerArg, ActionScalerArg,
    RewardScalerArg
)
from d3rlpy.constants import ActionSpace
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.models.q_functions import QFunctionFactory

from replay.models.sdac.policies import GumbelPolicy
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, hard_sync, torch_api, train_api


def create_gumbel_policy(
        observation_shape: Sequence[int],
        action_size: int,
        encoder_factory: EncoderFactory,
        gumb_temp = 1.0, 
        dist_tresh = 0.5
) -> GumbelPolicy:
    encoder = encoder_factory.create(observation_shape)
    return GumbelPolicy(encoder, action_size, gumbel_temp = gumb_temp, dist_tresh = dist_tresh)


class SDACImpl(SACImpl):
    _policy: Optional[GumbelPolicy]
    _targ_policy: Optional[GumbelPolicy]
        
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        gumb_temp=1.0, 
        dist_tresh=0.5
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            temp_learning_rate = temp_learning_rate,
            temp_optim_factory = temp_optim_factory,
            initial_temperature = initial_temperature
        )
        self.gumb_temp = gumb_temp, 
        self.dist_tresh = dist_tresh

    def _build_actor(self) -> None:
        self._policy = create_gumbel_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            gumb_temp = self.gumb_temp, 
            dist_tresh = self.dist_tresh
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
            keepdims = True
            return (target - entropy).mean(dim=1).reshape(-1,1)


class SDAC(SAC):
    r"""Soft Actor-Critic algorithm.

        SAC is a DDPG-based maximum entropy RL algorithm, which produces
        state-of-the-art performance in online RL settings.
        SAC leverages twin Q functions proposed in TD3. Additionally,
        `delayed policy update` in TD3 is also implemented, which is not done in
        the paper.

        .. math::

            L(\theta_i) = \mathbb{E}_{s_t,\, a_t,\, r_{t+1},\, s_{t+1} \sim D,\,
                                       a_{t+1} \sim \pi_\phi(\cdot|s_{t+1})} \Big[
                \big(y - Q_{\theta_i}(s_t, a_t)\big)^2\Big]

        .. math::

            y = r_{t+1} + \gamma \Big(\min_j Q_{\theta_j}(s_{t+1}, a_{t+1})
                - \alpha \log \big(\pi_\phi(a_{t+1}|s_{t+1})\big)\Big)

        .. math::

            J(\phi) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
                \Big[\alpha \log (\pi_\phi (a_t|s_t))
                  - \min_i Q_{\theta_i}\big(s_t, \pi_\phi(a_t|s_t)\big)\Big]

        The temperature parameter :math:`\alpha` is also automatically adjustable.

        .. math::

            J(\alpha) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
                \bigg[-\alpha \Big(\log \big(\pi_\phi(a_t|s_t)\big) + H\Big)\bigg]

        where :math:`H` is a target
        entropy, which is defined as :math:`\dim a`.

        References:
            * `Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep
              Reinforcement Learning with a Stochastic Actor.
              <https://arxiv.org/abs/1801.01290>`_
            * `Haarnoja et al., Soft Actor-Critic Algorithms and Applications.
              <https://arxiv.org/abs/1812.05905>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
            Q function factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        initial_temperature (float): initial temperature value.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.sac_impl.SDACImpl): algorithm implementation.

        """

    _impl: Optional[SDACImpl]

    def __init__(
            self,
            *,
            actor_learning_rate: float = 3e-4,
            critic_learning_rate: float = 3e-4,
            temp_learning_rate: float = 3e-4,
            actor_optim_factory: OptimizerFactory = AdamFactory(),
            critic_optim_factory: OptimizerFactory = AdamFactory(),
            temp_optim_factory: OptimizerFactory = AdamFactory(),
            actor_encoder_factory: EncoderArg = "default",
            critic_encoder_factory: EncoderArg = "default",
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
            impl: Optional[SDACImpl] = None,
            gumb_temp: float = 1.0, 
            dist_tresh: float = 0.5,
            **kwargs: Any
    ):
        super().__init__(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
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
            kwargs=kwargs,
        )
        self._impl = impl
        
        raise Exception(gumb_temp, dist_tresh)
        self.gumb_temp=gumb_temp, 
        self.dist_tresh=dist_tresh

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = SDACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            temp_learning_rate=self._temp_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            temp_optim_factory=self._temp_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            initial_temperature=self._initial_temperature,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            gumb_temp = self.gumb_temp,
            dist_tresh = self.dist_tresh
        )
        self._impl.build()

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

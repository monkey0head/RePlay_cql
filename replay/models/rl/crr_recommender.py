"""
Using CQL implementation from `d3rlpy` package.
For 'alpha' version PySpark DataFrame are converted to Pandas
"""

import d3rlpy.algos.crr as CRR_d3rlpy
from d3rlpy.argument_utility import (
    EncoderArg, QFuncArg, UseGPUArg, ScalerArg, ActionScalerArg,
    RewardScalerArg
)
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from pyspark.sql import DataFrame

from replay.models.rl.rl_recommender import RLRecommender


class CRRRecommender(RLRecommender):
    r"""Critic Reguralized Regression algorithm.

    CRR is a simple offline RL method similar to AWAC.

    The policy is trained as a supervised regression.

    .. math::

        J(\phi) = \mathbb{E}_{s_t, a_t \sim D}
            [\log \pi_\phi(a_t|s_t) f(Q_\theta, \pi_\phi, s_t, a_t)]

    where :math:`f` is a filter function which has several options. The first
    option is ``binary`` function.

    .. math::

        f := \mathbb{1} [A_\theta(s, a) > 0]

    The other is ``exp`` function.

    .. math::

        f := \exp(A(s, a) / \beta)

    The :math:`A(s, a)` is an average function which also has several options.
    The first option is ``mean``.

    .. math::

        A(s, a) = Q_\theta (s, a) - \frac{1}{m} \sum^m_j Q(s, a_j)

    The other one is ``max``.

    .. math::

        A(s, a) = Q_\theta (s, a) - \max^m_j Q(s, a_j)

    where :math:`a_j \sim \pi_\phi(s)`.

    In evaluation, the action is determined by Critic Weighted Policy (CWP).
    In CWP, the several actions are sampled from the policy function, and the
    final action is re-sampled from the estimated action-value distribution.

    References:
        * `Wang et al., Critic Reguralized Regression.
          <https://arxiv.org/abs/2006.15134>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
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
        beta (float): temperature value defined as :math:`\beta` above.
        n_action_samples (int): the number of sampled actions to calculate
            :math:`A(s, a)` and for CWP.
        advantage_type (str): advantage function type. The available options
            are ``['mean', 'max']``.
        weight_type (str): filter function type. The available options
            are ``['binary', 'exp']``.
        max_weight (float): maximum weight for cross-entropy loss.
        n_critics (int): the number of Q functions for ensemble.
        target_update_type (str): target update type. The available options are
            ``['hard', 'soft']``.
        tau (float): target network synchronization coefficiency used with
            ``soft`` target update.
        update_actor_interval (int): interval to update policy function used
            with ``hard`` target update.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.crr_impl.CRRImpl): algorithm implementation.

    """

    model: CRR_d3rlpy.CRR

    _search_space = {
        "actor_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "critic_learning_rate": {"type": "loguniform", "args": [3e-5, 3e-4]},
        "n_epochs": {"type": "int", "args": [3, 20]},
        "temp_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "alpha_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "gamma": {"type": "loguniform", "args": [0.9, 0.999]},
        "n_critics": {"type": "int", "args": [2, 4]},
    }
    
    def __init__(
            self, *,
            top_k: int, n_epochs: int = 1,
            action_randomization_scale: float = 0.,
            use_negative_events: bool = False,
            rating_based_reward: bool = False,
            reward_top_k: bool = False,
            test_log: DataFrame = None, 

            # CQL inner params
            actor_learning_rate: float = 3e-4,
            critic_learning_rate: float = 3e-4,
            actor_optim_factory: OptimizerFactory = AdamFactory(),
            critic_optim_factory: OptimizerFactory = AdamFactory(),
            actor_encoder_factory: EncoderArg = "default",
            critic_encoder_factory: EncoderArg = "default",
            q_func_factory: QFuncArg = "mean",
            batch_size: int = 100,
            n_frames: int = 1,
            n_steps: int = 1,
            gamma: float = 0.99,
            beta: float = 1.0,
            n_action_samples: int = 4,
            advantage_type: str = "mean",
            weight_type: str = "exp",
            max_weight: float = 20.0,
            n_critics: int = 1,
            target_update_type: str = "hard",
            tau: float = 5e-3,
            target_update_interval: int = 100,
            update_actor_interval: int = 1,
            use_gpu: UseGPUArg = False,
            scaler: ScalerArg = None,
            action_scaler: ActionScalerArg = None,
            reward_scaler: RewardScalerArg = None,
            **params
    ):
        model = CRR_d3rlpy.CRR(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            beta=beta,
            n_action_samples=n_action_samples,
            advantage_type=advantage_type,
            weight_type=weight_type,
            max_weight=max_weight,
            n_critics=n_critics,
            target_update_type=target_update_type,
            tau=tau,
            target_update_interval=target_update_interval,
            update_actor_interval=update_actor_interval,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            **params
        )

        super(CRRRecommender, self).__init__(
            model=model,
            test_log=test_log,
            top_k=top_k, n_epochs=n_epochs,
            action_randomization_scale=action_randomization_scale,
            use_negative_events=use_negative_events,
            rating_based_reward=rating_based_reward,
            reward_top_k=reward_top_k,
        )

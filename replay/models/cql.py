"""
Using CQL implementation from `d3rlpy` package.
For 'alpha' version PySpark DataFrame are converted to Pandas
"""

from typing import Optional, Callable

import d3rlpy.algos.cql as CQL_d3rlpy
import numpy as np
import pandas as pd
from d3rlpy.argument_utility import (
    EncoderArg, QFuncArg, UseGPUArg, ScalerArg, ActionScalerArg,
    RewardScalerArg
)
from d3rlpy.base import LearnableBase
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from pyspark.sql import functions as sf, DataFrame

from replay.data_preparator import DataPreparator
from replay.models import Recommender


class RLRecommender(Recommender):
    top_k: int
    n_epochs: int
    action_randomization_scale: float
    use_negative_events: bool
    rating_based_reward: bool
    rating_actions: bool
    reward_top_k: bool

    model: LearnableBase

    def __init__(
            self, *,
            model: LearnableBase,
            top_k: int,
            n_epochs: int = 1,
            action_randomization_scale: float = 0.,
            use_negative_events: bool = False,
            rating_based_reward: bool = False,
            rating_actions: bool = False,
            reward_top_k: bool = True
    ):
        super().__init__()
        self.model = model
        self.k = top_k
        self.n_epochs = n_epochs
        self.action_randomization_scale = action_randomization_scale
        self.use_negative_events = use_negative_events
        self.rating_based_reward = rating_based_reward
        self.rating_actions = rating_actions
        self.reward_top_k = reward_top_k

        self.train = None
        self.fitter = None

    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        if user_features or item_features:
            message = f'RL recommender does not support user/item features'
            self.logger.debug(message)

        users = users.toPandas().to_numpy().flatten()
        items = items.toPandas().to_numpy().flatten()

        # TODO: rewrite to applyInPandas with predictUserPairs parallel batch execution
        user_predictions = []
        for user in users:
            user_item_pairs = pd.DataFrame({
                'user_idx': np.repeat(user, len(items)),
                'item_idx': items
            })
            user_item_pairs['relevance'] = self.model.predict(user_item_pairs.to_numpy())
            user_predictions.append(user_item_pairs)

        prediction = pd.concat(user_predictions)

        # it doesn't explicitly filter seen items and doesn't return top k items
        # instead, it keeps all predictions as is to be filtered further by base methods
        return DataPreparator.read_as_spark_df(prediction)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        if self.train is None:
            self.train: MDPDataset = self._prepare_data(log)

        if self.fitter is None:
            self.fitter = self.model.fitter(self.train, n_epochs=self.n_epochs)

        try:
            next(self.fitter)
        except StopIteration:
            pass

    raw_rating_to_reward_rescale = {
        1.0: -1.0,
        2.0: -0.3,
        3.0: 0.25,
        4.0: 0.7,
        5.0: 1.0,
    }
    binary_rating_to_reward_rescale = {
        1.0: -1.0,
        2.0: -1.0,
        3.0: 1.0,
        4.0: 1.0,
        5.0: 1.0,
    }

    def _prepare_data(self, log: DataFrame) -> MDPDataset:
        if not self.use_negative_events:
            # remove negative events
            log = log.filter(sf.col('relevance') >= sf.lit(3.0))

        # TODO: consider making calculations in Spark before converting to pandas
        user_logs = log.toPandas().sort_values(['user_idx', 'timestamp'], ascending=True)

        if self.rating_based_reward:
            rescale = self.raw_rating_to_reward_rescale
        else:
            rescale = self.binary_rating_to_reward_rescale
        rewards = user_logs['relevance'].map(rescale).to_numpy()

        if self.reward_top_k:
            # additionally reward top-K watched movies
            user_top_k_idxs = (
                user_logs
                .sort_values(['relevance', 'timestamp'], ascending=[False, True])
                .groupby('user_idx')
                .head(self.k)
                .index
            )
            # rescale positives and additionally reward top-K watched movies
            rewards[rewards > 0] /= 2
            rewards[user_top_k_idxs] += 0.5

        # every user has his own episode (the latest item is defined as terminal)
        user_terminal_idxs = (
            user_logs[::-1]
            .groupby('user_idx')
            .head(1)
            .index
        )
        terminals = np.zeros(len(user_logs))
        terminals[user_terminal_idxs] = 1

        actions = user_logs['relevance'].to_numpy()
        if not self.rating_actions:
            actions = (actions >= 3).astype(int)

        if self.action_randomization_scale > 0:
            # cannot set zero scale as d3rlpy will treat transitions as discrete :/
            action_randomization_scale = self.action_randomization_scale
            action_randomization = np.random.randn(len(user_logs)) * action_randomization_scale
            actions += action_randomization

        train_dataset = MDPDataset(
            observations=np.array(user_logs[['user_idx', 'item_idx']]),
            actions=actions[:, None],
            rewards=rewards,
            terminals=terminals
        )
        return train_dataset

    @property
    def _init_args(self):
        args = dict(
            top_k=self.top_k,
            n_epochs=self.n_epochs,
            action_randomization_scale=self.action_randomization_scale,
            use_negative_events=self.use_negative_events,
            rating_based_reward=self.rating_based_reward,
            reward_top_k=self.reward_top_k
        )
        args.update(**self.model.get_params())
        return args


class CQL(RLRecommender):
    r"""Conservative Q-Learning algorithm.

    CQL is a SAC-based data-driven deep reinforcement learning algorithm, which
    achieves state-of-the-art performance in offline RL problems.

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::

        L(\theta_i) = \alpha\, \mathbb{E}_{s_t \sim D}
            \left[\log{\sum_a \exp{Q_{\theta_i}(s_t, a)}}
             - \mathbb{E}_{a \sim D} \big[Q_{\theta_i}(s_t, a)\big] - \tau\right]
            + L_\mathrm{SAC}(\theta_i)

    where :math:`\alpha` is an automatically adjustable value via Lagrangian
    dual gradient descent and :math:`\tau` is a threshold value.
    If the action-value difference is smaller than :math:`\tau`, the
    :math:`\alpha` will become smaller.
    Otherwise, the :math:`\alpha` will become larger to aggressively penalize
    action-values.

    In continuous control, :math:`\log{\sum_a \exp{Q(s, a)}}` is computed as
    follows.

    .. math::

        \log{\sum_a \exp{Q(s, a)}} \approx \log{\left(
            \frac{1}{2N} \sum_{a_i \sim \text{Unif}(a)}^N
                \left[\frac{\exp{Q(s, a_i)}}{\text{Unif}(a)}\right]
            + \frac{1}{2N} \sum_{a_i \sim \pi_\phi(a|s)}^N
                \left[\frac{\exp{Q(s, a_i)}}{\pi_\phi(a_i|s)}\right]\right)}

    where :math:`N` is the number of sampled actions.

    An implementation of this algorithm is heavily based on the corresponding implementation
    in the d3rlpy library (see https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/cql.py)

    The rest of optimization is exactly same as :class:`d3rlpy.algos.SAC`.

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float):
            learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the temperature.
        alpha_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for :math:`\alpha`.
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
        initial_alpha (float): initial :math:`\alpha` value.
        alpha_threshold (float): threshold value described as :math:`\tau`.
        conservative_weight (float): constant weight to scale conservative loss.
        n_action_samples (int): the number of sampled actions to compute
            :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): flag to use SAC-style backup.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.cql_impl.CQLImpl): algorithm implementation.

    """

    model: CQL_d3rlpy.CQL

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
            rating_actions: bool = False,
            reward_top_k: bool = False,

            # CQL inner params
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 3e-4,
            temp_learning_rate: float = 1e-4,
            alpha_learning_rate: float = 1e-4,
            actor_optim_factory: OptimizerFactory = AdamFactory(),
            critic_optim_factory: OptimizerFactory = AdamFactory(),
            temp_optim_factory: OptimizerFactory = AdamFactory(),
            alpha_optim_factory: OptimizerFactory = AdamFactory(),
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
            initial_alpha: float = 1.0,
            alpha_threshold: float = 10.0,
            conservative_weight: float = 5.0,
            n_action_samples: int = 10,
            soft_q_backup: bool = False,
            use_gpu: UseGPUArg = False,
            scaler: ScalerArg = None,
            action_scaler: ActionScalerArg = None,
            reward_scaler: RewardScalerArg = None,
            **params
    ):
        model = CQL_d3rlpy.CQL(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            alpha_learning_rate=alpha_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            alpha_optim_factory=alpha_optim_factory,
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
            initial_alpha=initial_alpha,
            alpha_threshold=alpha_threshold,
            conservative_weight=conservative_weight,
            n_action_samples=n_action_samples,
            soft_q_backup=soft_q_backup,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            **params
        )

        super(CQL, self).__init__(
            model=model,
            top_k=top_k, n_epochs=n_epochs,
            action_randomization_scale=action_randomization_scale,
            use_negative_events=use_negative_events,
            rating_based_reward=rating_based_reward,
            rating_actions=rating_actions,
            reward_top_k=reward_top_k,
        )

from rs_datasets import MovieLens
from d3rlpy.base import LearnableBase
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from pyspark.sql import functions as sf, DataFrame
import numpy as np
from typing import Optional, Callable
import pandas as pd
from d3rlpy.metrics.scorer import evaluate_on_environment
from replay.models.sdac.sdac_impl import SDAC
from replay.models.cql import CQL
from fake_recommender_env import FakeRecomenderEnv
from d3rlpy.models.torch.encoders import _VectorEncoder, EncoderWithAction
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence
import d3rlpy
import torch
import torch.nn.functional as F
from torch import nn

def _prepare_data(log: DataFrame) -> MDPDataset:
        use_negative_events = True #False
        rating_based_reward = False #False
        reward_top_k = True
        k = 10
        
        test_size = 0.3
        action_randomization_scale = 0.01
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
        if not use_negative_events:
            # remove negative events
            log = log[log['rating'] >= 3]

        # TODO: consider making calculations in Spark before converting to pandas
        user_logs = log.sort_values(['user_id', 'timestamp'], ascending=True)

        if rating_based_reward:
            rescale = raw_rating_to_reward_rescale
        else:
            rescale = binary_rating_to_reward_rescale
        rewards = user_logs['rating'].map(rescale).to_numpy()

        if reward_top_k:
            # additionally reward top-K watched movies
            user_top_k_idxs = (
                
                user_logs
                .sort_values(['rating', 'timestamp'], ascending=[False, True])
                .groupby('user_id')
                .head(k)
                .index
            )
            # rescale positives and additionally reward top-K watched movies
            rewards[rewards > 0] /= 2
            rewards[user_top_k_idxs] += 0.5

        user_logs['rewards'] = rewards

        # every user has his own episode (the latest item is defined as terminal)
        user_terminal_idxs = (
            user_logs[::-1]
            .groupby('user_id')
            .head(1)
            .index
        )
        terminals = np.zeros(len(user_logs))
        terminals[user_terminal_idxs] = 1
        user_logs['terminals'] = terminals

        # cannot set zero scale as d3rlpy will treat transitions as discrete :/
        
        
        #разбиение на трейн тест
        user_id_list = list(set(user_logs['user_id']))
        count_of_test = int(test_size*len(user_id_list))
        test_idx = int(user_id_list[-count_of_test])
        
        user_logs_train = user_logs[user_logs['user_id'].astype(int) < test_idx]
        user_logs_test = user_logs[user_logs['user_id'].astype(int) >= test_idx]
        
        action_randomization_scale = action_randomization_scale + 1e-4
        action_randomization = np.random.randn(len(user_logs_train)) * action_randomization_scale

        train_dataset = MDPDataset(
            observations=np.array(user_logs_train[['user_id', 'item_id']]),
            actions=np.array(
                user_logs_train['rating']
            )[:, None] ,
            rewards=user_logs_train['rewards'],
            terminals=user_logs_train['terminals']
        )
      #  print( user_logs_test['rating'])
        test_dataset = MDPDataset(
            observations=np.array(user_logs_test[['user_id', 'item_id']]),
            actions=np.array(
                user_logs_test['rating'] 
            )[:, None],
            rewards=user_logs_test['rewards'],
            terminals=user_logs_test['terminals']
        )
        return train_dataset, user_logs_train
        

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

  #  def create(self, observation_shape):
   #     return CustomEncoder(observation_shape, self.feature_size)

    def create_with_action(self, observation_shape, action_size, discrete_action):
        return VectorEncoderWithAction(observation_shape, action_size, self.feature_size)

    def get_params(self, deep=False):
        return {"feature_size": self.feature_size}
        
if __name__ == "__main__":
	ds = MovieLens(version="1m")
	train_dataset,user_logs_train = _prepare_data(ds.ratings)
	#encoder_factory=CustomEncoderFactory(64)
	sdac = SDAC(use_gpu=False, encoder_factory=CustomEncoderFactory(64))
	env = FakeRecomenderEnv(user_logs_train[:1000], 10)
	evaluate_scorer = evaluate_on_environment(env)
	sdac.fit(train_dataset,
        eval_episodes=train_dataset,
        n_epochs=10,
        scorers={'environment': evaluate_scorer})
        
 

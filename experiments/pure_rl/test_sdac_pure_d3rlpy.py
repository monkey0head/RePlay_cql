from rs_datasets import MovieLens
from d3rlpy.base import LearnableBase
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from pyspark.sql import functions as sf, DataFrame
import numpy as np
from typing import Optional, Callable
import pandas as pd
from d3rlpy.metrics.scorer import evaluate_on_environment
from replay.models.sdac.sdac import SDAC
from replay.models.cql import CQL
from fake_recommender_env import FakeRecomenderEnv
    
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
        
if __name__ == "__main__":
	ds = MovieLens(version="1m")
	train_dataset,user_logs_train = _prepare_data(ds.ratings)
	
	sdac = SDAC(use_gpu=True)
	env = FakeRecomenderEnv(user_logs_train[:1000], 10)
	evaluate_scorer = evaluate_on_environment(env)
	sdac.fit(train_dataset,
        eval_episodes=train_dataset,
        n_epochs=10,
        scorers={'environment': evaluate_scorer})
        
 

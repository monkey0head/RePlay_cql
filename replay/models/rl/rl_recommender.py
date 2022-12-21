from typing import Optional

import numpy as np
import pandas as pd
from d3rlpy.base import LearnableBase
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import evaluate_on_environment
from pyspark.sql import DataFrame, functions as sf

from replay.data_preparator import DataPreparator
from replay.models.base_rec import Recommender
from replay.models.rl.fake_recommender_env import FakeRecomenderEnv

def random_embeddings(df, emb_size):
    mapping = dict()
    inv_mapping = dict()
    users = list(set(df))
    for user in users:
        new_vector = np.random.uniform(0, 1, size=emb_size)
        #new_vector = np.ones(emb_size)
        mapping[user] = tuple(new_vector.tolist())
        inv_mapping[tuple(new_vector)] = user
    return mapping, inv_mapping

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
            test_log: DataFrame,
            n_epochs: int = 1,
            action_randomization_scale: float = 0.,
            use_negative_events: bool = False,
            rating_based_reward: bool = False,
            rating_actions: bool = False,
            reward_top_k: bool = True,

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
        self.mapping_items = None
        self.mapping_users = None
        self.inv_mapp_items = None
        self.inv_mapp_users = None
        self.train = None
        self.fitter = None
        self.test_log = test_log

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
        
        mapping_users = list(self.mapping_users.keys())
        mapping_items = list(self.mapping_items.keys())
        users = [self.mapping_users[user] if user in mapping_users 
                 else np.random.uniform(0, 1, size=8) 
                 for user in users ]
        
        items = [self.mapping_items[item] if item in mapping_items 
                 else np.random.uniform(0, 1, size=8) 
                 for item in items ]
                 
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
        if self.test_log:
            _, val_df = self._prepare_data(self.test_log, True)
           # raise Exception (len(self.test_log))
            indx = np.arange(len(val_df))
            np.random.shuffle(indx)
            env = FakeRecomenderEnv(val_df.iloc[indx[:10000]], self.k)
            # evaluate_scorer = evaluate_on_environment(env)

        if self.fitter is None:
            self.fitter = self.model.fitter(
                self.train,
                n_epochs=self.n_epochs,
                # eval_episodes=self.train,
                # scorers={'environment': evaluate_scorer}
            )

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
    
    def _idx2obs(self, item_user_array):
       # observations = np.array(user_logs[['user_idx', 'item_idx']])
        observations = []
        for obs in item_user_array:
            user_emb = self.mapping_users[obs[0]]
            item_emb = self.mapping_items[obs[1]]
            
            new_obs = list(user_emb) + list(item_emb)
            observations.append(new_obs)
        return np.asarray(observations)


    def _prepare_data(self, log: DataFrame, return_pd_df = False) -> MDPDataset:
        if not self.use_negative_events:
            # remove negative events
            log = log.filter(sf.col('relevance') >= sf.lit(3.0))

        # TODO: consider making calculations in Spark before converting to pandas
        user_logs = log.toPandas().sort_values(['user_idx', 'timestamp'], ascending=True)
        
        self.mapping_items, self.inv_mapp_items = random_embeddings(user_logs['item_idx'], emb_size = 8)
        self.mapping_users, self.inv_mapp_users = random_embeddings(user_logs['user_idx'], emb_size = 8)
        
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
            actions = actions.astype(np.float64)
            actions += action_randomization
            
        observations = self._idx2obs(np.array(user_logs[['user_idx', 'item_idx']]))
       # print(np.asarray(observations[:2]))
       # print("-------------------------------------")
       # print(observations.shape)
        train_dataset = MDPDataset(
            observations=observations,
            actions=actions[:, None],
            rewards=rewards,
            terminals=terminals
        )
        if return_pd_df:
            user_logs['rating'] = actions
            user_logs['rewards'] = rewards
            user_logs['terminals'] = terminals
            return train_dataset, user_logs
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

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
from tqdm import tqdm

from replay.models.rl.embeddings import random_embeddings, als_embeddings, ddpg_embeddings
from replay.models.rl.rating_mdp.rating_metrics.metrics import true_ndcg
from replay.models.rl.rating_mdp.data_preparing.prepare_data import item_user_pair

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
        self.user_item_pairs = None
        self.observations_test = None
        self.test_log_pd = None
        
    def _idx2obs(self, item_user_array, show_logs = True):
       # observations = np.array(user_logs[['user_idx', 'item_idx']])
        observations = []
        if show_logs: print("Prepare embedings...")
        out_of_emb_users = 0
        out_of_emb_items = 0
        if show_logs:
            gen = tqdm(item_user_array)
        else:
            gen = item_user_array
        for obs in gen:
            if obs[0] in list(self.mapping_users.keys()):
                user_emb = self.mapping_users[obs[0]]
            else:
              #  print(obs[0])
                out_of_emb_users += 1
                user_emb = np.random.uniform(0, 1, size=8)
            
            if obs[1] in list(self.mapping_items.keys()):
                item_emb = self.mapping_items[obs[1]]
            else:
               # print(obs[1])
                out_of_emb_items += 1
                item_emb = np.random.uniform(0, 1, size=8)
            
            new_obs = list(user_emb) + list(item_emb)
            observations.append(new_obs)
        if show_logs:
            print(f"Out of embeddings users {out_of_emb_users}/{len(item_user_array)}, items {out_of_emb_items}/{len(item_user_array)}. \n")
        return np.asarray(observations)
    
    def __make_obs_for_test(self, users, items):
        self.user_item_pairs = []
        self.observations_test = []
        for user in users:
            user_item_pairs = pd.DataFrame({
                'user_idx': np.repeat(user, len(items)),
                'item_idx': items
            })            
            observation =  self._idx2obs(user_item_pairs.to_numpy(), show_logs = False)
            self.user_item_pairs.append(user_item_pairs)
            self.observations_test.append(observation)
            
        
    
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
        if self.observations_test is None:
            self.__make_obs_for_test(users, items)
            
        for user_item_pairs,observation in  zip(self.user_item_pairs, self.observations_test):
            user_item_pairs['relevance'] = self.model.predict(observation)
            print(user_item_pairs['relevance'])
            user_predictions.append(user_item_pairs)

        prediction = pd.concat(user_predictions)
        
        print(prediction['relevance'])
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
            #user_logs = log.toPandas().sort_values(['user_idx', 'timestamp'], ascending=True)
            if self.test_log_pd is None:
                self.test_log_pd = self.test_log.toPandas().sort_values(['user_idx', 'timestamp'], ascending=True)
            items_obs_orig = np.unique(self.test_log_pd['item_idx'].values)
            users_obs_orig = np.unique(self.test_log_pd['user_idx'].values)
            
            #print(self.mapping_items.keys())
            items_obs = [self.mapping_items[item] for item in items_obs_orig if item in list(items_obs_orig.keys())]
            users_obs = [self.mapping_users[user] for user in users_obs_orig if user in list(users_obs_orig.keys())]
            
            print(len(items_obs),"/", len(items_obs_orig))
            print(len(users_obs),"/", len(users_obs_orig))
            
            
            obs_for_pred, users = item_user_pair(items_obs, users_obs)    
            self.scorer = true_ndcg(obs_for_pred, users, self.inv_mapp_items, top_k = 10)
        
        if self.test_log:
            test_mdp, val_df = self._prepare_data(self.test_log, True)
            indx = np.arange(len(val_df))
            np.random.shuffle(indx)
            #env = FakeRecomenderEnv(val_df.iloc[indx[:10000]], self.k)
        #evaluate_on_environment(env)

        if self.fitter is None:
            self.fitter = 3
            self.fitter = self.model.fitter(
                self.train,
                n_epochs=self.n_epochs,
                #n_steps = 2000*self.n_epochs,
               # n_steps_per_epoch = 2000,
                eval_episodes=test_mdp,
                scorers={'ndcg_sorer': self.scorer}
            )
            
           # self.model.fit(self.train, eval_episodes=self.train,n_epochs = 10, scorers={'NDCG': self.scorer})

        try:
        #    print(len(self.user_logs))
          #  print(len(self.train))
          ##  print(self.train.observations[0])
          #  print(self.train.actions)
          #  print(self.train.rewards)
          #  self.model.fit(self.train, eval_episodes=self.train,n_epochs = 100, scorers={'NDCG': self.scorer})
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


    def _prepare_data(self, log: DataFrame, return_pd_df = False) -> MDPDataset:
        if not self.use_negative_events:
            # remove negative events
            log = log.filter(sf.col('relevance') >= sf.lit(3.0))

        # TODO: consider making calculations in Spark before converting to pandas
        user_logs = log.toPandas().sort_values(['user_idx', 'timestamp'], ascending=True)
        
        if self.mapping_items is None:
            print("! ---- Generate new embedings ---- !")
            self.user_logs = user_logs
            print(self.user_logs)
            embedings = als_embeddings(user_logs, emb_size = 8)
            self.mapping_users, self.inv_mapp_users, self.mapping_items, self.inv_mapp_items = embedings
          #  self.mapping_users, self.inv_mapp_users = als_embeddings(user_logs, emb_size = 8)
        
        #if self.rating_based_reward:
           # rescale = self.raw_rating_to_reward_rescale
        #else:
          #  rescale = self.binary_rating_to_reward_rescale
       # rewards = user_logs['relevance'].map(rescale).to_numpy()
        rewards = user_logs['relevance'].to_numpy()

#         if self.reward_top_k:
#             # additionally reward top-K watched movies
#             user_top_k_idxs = (
#                 user_logs
#                 .sort_values(['relevance', 'timestamp'], ascending=[False, True])
#                 .groupby('user_idx')
#                 .head(self.k)
#                 .index
#             )
#             # rescale positives and additionally reward top-K watched movies
#             rewards[rewards > 0] /= 2
#             rewards[user_top_k_idxs] += 0.5

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
       # if not self.rating_actions:
           # actions = (actions >= 3)#.astype(int)

       # if self.action_randomization_scale > 0:
            # cannot set zero scale as d3rlpy will treat transitions as discrete :/
           # action_randomization_scale = self.action_randomization_scale
          #  action_randomization = np.random.randn(len(user_logs)) * action_randomization_scale
          #  actions = actions.astype(np.float64)
          #  actions += action_randomization
            
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
        print(actions)
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

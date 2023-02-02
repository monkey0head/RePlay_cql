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
from replay.models.rl.rating_mdp.mdp.rating_mdp import negative_reward, mono_reward
from replay.models.rl.rating_mdp.data_preparing.prepare_data import item_user_pair



class RLRecommender(Recommender):
    top_k: int
    n_epochs: int
    model: LearnableBase

    def __init__(
            self, *,
            model: LearnableBase,
            top_k: int,
            test_log: DataFrame,
            n_epochs: int = 1,
            reward_function: str

    ):
        super().__init__()
        self.model = model
        self.k = top_k
        self.n_epochs = n_epochs
        self.reward_function = negative_reward if reward_function=="neg" else mono_reward
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
                out_of_emb_users += 1
                user_emb = np.random.uniform(0, 1, size=8)
            
            if obs[1] in list(self.mapping_items.keys()):
                item_emb = self.mapping_items[obs[1]]
            else:
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
           # print(observation)
            #exit()
            user_item_pairs['relevance'] = [0 if pred <3 else 1 for pred in self.model.predict(observation)]
          #  user_item_pairs_cp = user_item_pairs.copy()
           # ones = user_item_pairs['relevance'] >= 3
           # zeros = user_item_pairs['relevance'] < 3
           # user_item_pairs_cp[zeros]['relevance'] = 0
           # user_item_pairs_cp[ones]['relevance'] = 1
            
           	#user_item_pairs = user_item_pairs.sort_values(by='relevance', ascending=True)#[:100]
           # print(user_item_pairs)
           # exit()
            user_predictions.append(user_item_pairs)

        prediction = pd.concat(user_predictions)
        print(prediction)
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
                
            # Make scorer for wandb
            if self.test_log_pd is None:
                self.test_log_pd = self.test_log.toPandas().sort_values(['user_idx', 'timestamp'], ascending=True)
            items_obs_orig = np.unique(self.test_log_pd['item_idx'].values)
            users_obs_orig = np.unique(self.test_log_pd['user_idx'].values)
            items_obs = [self.mapping_items[item] for item in items_obs_orig if item in list(self.mapping_items.keys())]
            users_obs = [self.mapping_users[user] for user in users_obs_orig if user in list(self.mapping_users.keys())]      
            obs_for_pred, users = item_user_pair(items_obs, users_obs)    
            self.scorer = true_ndcg(obs_for_pred, users, self.inv_mapp_items, top_k = 10)
        
        if self.test_log:
            test_mdp, val_df = self._prepare_data(self.test_log_pd, return_pd_df = True, already_pd = True)
            indx = np.arange(len(val_df))
            np.random.shuffle(indx)

        if self.fitter is None:
            self.fitter = 3
            self.fitter = self.model.fitter(
               self.train,
               n_epochs=self.n_epochs,
               eval_episodes=test_mdp,
               scorers={'ndcg_sorer': self.scorer}
            )
        try:
            next(self.fitter)
        except StopIteration:
            pass


    def _prepare_data(self, log: DataFrame, return_pd_df = False, 
                      is_test_run = False, already_pd = False) -> MDPDataset:
        
        # TODO: consider making calculations in Spark before converting to pandas  
        if already_pd:
            user_logs = log
        else:
            user_logs = log.toPandas().sort_values(['user_idx', 'timestamp'], ascending=True)
        if self.mapping_items is None:
            print("! ---- Generate new embedings ---- !")
            self.user_logs = user_logs
            embedings = als_embeddings(user_logs, emb_size = 8)
            self.mapping_users, self.inv_mapp_users, self.mapping_items, self.inv_mapp_items = embedings

        # every user has his own episode (the latest item is defined as terminal)
        user_terminal_idxs = (
            user_logs[::-1]
            .groupby('user_idx')
            .head(1)
            .index
        )
        
        # Make MDP
        terminals = np.zeros(len(user_logs))
        terminals[user_terminal_idxs] = 1            
        observations = self._idx2obs(np.array(user_logs[['user_idx', 'item_idx']]))   
        
        # If it is test run augment observations
        if not is_test_run:            
            rewards, actions = self.reward_function(user_logs, invert = False, rating_column='relevance')
        else:
            rewards, actions = self.reward_function(user_logs, invert = True, rating_column='relevance')
            observations = np.append(observations, observations, axis = 0)
            terminals = np.append(terminals, terminals, axis = 0)
        
        train_dataset = MDPDataset(
            observations= np.asarray(observations),
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
            reward_function = self.reward_function
        )
        args.update(**self.model.get_params())
        return args

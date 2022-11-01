import math  
import gym
import numpy as np
from gym.spaces import Discrete, Box, Tuple
import wandb
import pandas as pd

def ndcg(k, pred, ground_truth) -> float:
        pred_len = min(k, len(pred))
        ground_truth_len = min(k, len(ground_truth))
        denom = [1 / math.log2(i + 2) for i in range(k)]
        dcg = sum(denom[i] for i in range(pred_len) if pred[i] in ground_truth)
        idcg = sum(denom[:ground_truth_len])

        return dcg / idcg
    
def mape(k, pred, ground_truth) -> float:
        length = min(k, len(pred))
        max_good = min(k, len(ground_truth))
        if len(ground_truth) == 0 or len(pred) == 0:
            return 0
        tp_cum = 0
        result = 0
        for i in range(length):
            if pred[i] in ground_truth:
                tp_cum += 1
                result += tp_cum / ((i + 1) * max_good)
        return result      
        

def original_for_user(df, target, k = 10):
    mask = df['user_idx'] == target
    user_relevance = df[mask]
    return user_relevance.sort_values(['rating'])[::-1][:k]
   

class FakeRecomenderEnv(gym.Env):
    def __init__(self, test_data, top_k):
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = Box(0,100000, (2,))
        self.log_data = test_data
        self.top_k = top_k
        self.steps = 0
        self.episode_num = 0
        self.total_ndsg = []
        self.total_mape = []
       
        self.episodes = list(set(self.log_data['user_idx']))
        if len(self.episodes) == 0:
            raise Exception (self.log_data.keys())
        self.total_episodes = 0
        #mask = self.log_data['user_id'] == episodes[episode_num]
        self.current_episode = None
        self.run = 0
       

    def step(self, action): 
        #print(action)
        self.relevance_hist.append(action)
        done = False
        reward = 0
        ob = (self.current_episode['user_idx'].values[self.steps], 
                self.current_episode['item_idx'].values[self.steps])
        self.steps += 1
       
        	
        if len(self.current_episode['user_idx']) == self.steps:
           # done = True
          #  print(len(self.user_hist), len(self.item_hist), len(self.relevance_hist))
            pred_df = pd.DataFrame({'user_idx': self.user_hist, 'item_hist': self.item_hist,
                                    'relevance': self.relevance_hist})
            pred_top_k = pred_df.sort_values(['relevance'])[::-1][:self.top_k]
            ndcg_ = ndcg( self.top_k, pred_top_k['item_hist'].values, self.original['item_idx'].values)
            mape_ = mape( self.top_k, pred_top_k['item_hist'].values, self.original['item_idx'].values)
            
           # print(pred_top_k['item_hist'].values, self.original['item_idx'].values, ndcg_)
            wandb.log({"episode": self.total_episodes, "NDCG": ndcg_, "MAP": mape_})
            self.total_ndsg.append(ndcg_)
            self.total_mape.append(mape_)
            ob = []            
            
            if self.episode_num >= len(self.episodes)-1:
              done = True
              wandb.log({"run": self.run, "total_NDCG": np.mean(np.asarray(self.total_ndsg)), "total_MAP": np.mean(np.asarray(self.total_mape))})
              self.total_ndsg = []
              self.total_mape = []
              self.run += 1
              self.episode_num = 0
            ob = self.reset()
        else:
            self.user_hist.append(self.current_episode['user_idx'].values[self.steps])
            self.item_hist.append(self.current_episode['item_idx'].values[self.steps])    
         
        return np.asarray(ob), reward, done, {}
        
  #  def fake_reset():
    
    
    def reset(self):
        self.user_hist = []
        self.item_hist = []
        self.relevance_hist = []
        self.total_episodes += 1
        self.episode_num += 1
     #   if self.episode_num == len(self.episodes):
      #      self.episode_num = 0
        
        self.steps = 0 
        try:
        	mask = self.log_data['user_idx'] == self.episodes[self.episode_num]
        except Exception as e:
        	raise Exception(e, self.episode_num, self.episodes)
        self.current_episode = self.log_data[mask]
       # print(self.current_episode['user_id'])
        self.user_hist.append(self.current_episode['user_idx'].values[0])
        self.item_hist.append( self.current_episode['item_idx'].values[0])
        self.original = original_for_user(self.log_data, self.current_episode['user_idx'].values[0], k = self.top_k)
        obs = self.current_episode['user_idx'].values[0], \
                       self.current_episode['item_idx'].values[0]
      #  print( np.asarray(obs))
        return np.asarray(obs)



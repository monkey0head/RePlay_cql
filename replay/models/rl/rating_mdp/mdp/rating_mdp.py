import pandas as pd
import numpy as np

def binary_actions(df, invert = False):
    reward = df['rating'].to_numpy().astype(np.int)
    actions = np.zeros_like(reward)
    actions[reward >= 3] = 1
    return reward, actions


def original_actions(df, invert = False):
    reward = df['rating'].to_numpy().astype(np.int)
    actions = reward.copy()
    return reward, actions

def negative_reward_binary_actions(df, invert = False):
    reward = df['rating'].to_numpy().astype(np.int)
    actions = np.zeros_like(reward)
    actions[reward>=4] = 1
    reward[reward<4] = -1
    return reward, actions

def negative_reward(df, invert = False):
    reward = df['rating'].to_numpy().astype(np.int)
    actions = reward.copy()
    
  #  reward[reward<4] = -1
    reward[:] = 1
    
    if invert:

        actions_cp = actions.copy()
        actions_cp = np.abs(actions_cp - 6)
         
        reward = np.append(reward, reward_cp)
        
        reward_cp = -np.abs(actions - actions_cp)/10
        #reward_cp[:] = -1
        
        actions = np.append(actions, actions_cp)
    return reward, actions

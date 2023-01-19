import pandas as pd
import numpy as np

def binary_actions(df):
    reward = df['rating'].to_numpy().astype(np.int)
    actions = np.zeros_like(reward)
    actions[reward >= 3] = 1
    return reward, actions

def original_actions(df):
    reward = df['rating'].to_numpy().astype(np.int)
    actions = reward.copy()
    return reward, actions

def negative_reward_binary_actions(df):
    reward = df['rating'].to_numpy().astype(np.int)
    actions = np.zeros_like(reward)
    actions[reward>=4] = 1
    reward[reward<4] = -1
    return reward, actions

def negative_reward(df):
    reward = df['rating'].to_numpy().astype(np.int)
    actions = reward.copy()
    reward[reward<4] = -1
    return reward, actions

import pandas as pd
import numpy as np
from tqdm import tqdm

import d3rlpy
from d3rlpy.base import LearnableBase
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import evaluate_on_environment
#from d3rlpy.algos import DiscreteCQL, DiscreteSAC, SDAC
from replay.models.rl.rating_mdp.embeddings.embeddings import als_embeddings, random_embeddings, ddpg_embeddings
#from embeddings.embeddings import als_embeddings, random_embeddings, ddpg_embeddings

def reidification(df_full):
    users = df_full['user_id']
    items = df_full['item_id']
    values = df_full['rating']
    
    timestamp = df_full['timestamp']
    count_items = len(list(set(items)))
    count_users = len(list(set(users)))

    item_list = list(range(1,count_items+1))
    user_list = list(range(1,count_users+1))

    item_mapping = dict(zip(list(set(items)), item_list))
    user_mapping = dict(zip(list(set(users)), user_list))
    
    new_items = [item_mapping[item] for item in items]
    new_users = [user_mapping[user] for user in users]
    
    new_df = pd.DataFrame({'timestamp':timestamp, 'user_id':new_users, 'item_id':new_items, 'rating':values})
    return new_df
  
def _idx2obs(item_user_array, mapping_users, mapping_items, show_logs = True):
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
            if obs[0] in list(mapping_users.keys()):
                user_emb = mapping_users[obs[0]]
            else:
                out_of_emb_users += 1
                user_emb = np.random.uniform(0, 1, size=8)
            
            if obs[1] in list(mapping_items.keys()):
                item_emb = mapping_items[obs[1]]
            else:
                out_of_emb_items += 1
                item_emb = np.random.uniform(0, 1, size=8)
            
            new_obs = list(user_emb) + list(item_emb)
            observations.append(new_obs)
        if show_logs:
            print(f"Out of embeddings users {out_of_emb_users}/{len(item_user_array)},items {out_of_emb_items}/{len(item_user_array)}. \n")
        return np.asarray(observations)
    

def _prepare_data(user_logs, emb = True, return_pd_df = False, pfunc = None):
        user_logs = user_logs.sort_values(['user_id', 'timestamp'], ascending=True) 
        if emb == 'als':
            mapping_users, inv_mapp_users, mapping_items, inv_mapp_items = als_embeddings(user_logs)
        elif emb == 'rand':
            mapping_items, inv_mapp_items = random_embeddings(user_logs['item_id'], emb_size = 8)
            mapping_users, inv_mapp_users = random_embeddings(user_logs['user_id'], emb_size = 8)
        ###NEED TO REMOVE COSTIL with parametrs
        elif emb == 'ddpg':
            mapping_users, inv_mapp_users, mapping_items, inv_mapp_items = ddpg_embeddings(user_logs)
            
        # every user has his own episode (the latest item is defined as terminal)
        user_terminal_idxs = (
            user_logs[::-1]
            .groupby('user_id')
            .head(1)
            .index
        )
        mask_train = user_logs['dataset']=='train'
        mask_test = user_logs['dataset']=='test'
        
        user_logs_train = user_logs[mask_train]
        values, actions = pfunc(user_logs_train, invert = True)   
        observations = _idx2obs(np.array(user_logs_train[['user_id', 'item_id']]), mapping_users, mapping_items)
        observations = np.append(observations,observations,axis = 0) 
        user_terminal_idxs = (
            user_logs_train[::-1]
            .groupby('user_id')
            .head(1)
            .index
        )
        terminals = np.zeros(len(user_logs))
        terminals[user_terminal_idxs] = 1
        terminals = np.append(terminals,terminals,axis = 0) 
        
        print(observations.shape)
        print(actions.shape)
        print(values.shape)
        print(terminals.shape)
        
        train_dataset = MDPDataset(
            observations=observations,
            actions=actions[:, None],
            rewards=values,
            terminals=terminals
        )
        
        user_logs_test = user_logs[mask_test]
        values, actions = pfunc(user_logs_test)   
        observations = _idx2obs(np.array(user_logs_test[['user_id', 'item_id']]), mapping_users, mapping_items)
        
#         user_terminal_idxs = (
#             user_logs_test[::-1]
#             .groupby('user_id')
#             .head(1)
#             .index
#         )
#         terminals = np.zeros(len(user_logs_test))
#         terminals[user_terminal_idxs] = 1
                  
        test_dataset = MDPDataset(
            observations=observations,
            actions=actions[:, None],
            rewards=values,
            terminals=terminals
        )       
        return train_dataset,test_dataset, mapping_items, mapping_users, inv_mapp_items, inv_mapp_users, mask_test

def item_user_pair(items, users):  
     obs_for_pred = []
     users_full = []
     for user in users:
        for item in items:
            obs_zeros = np.zeros(16)
            obs_zeros[:8] = user
            obs_zeros[8:] = item
            obs_for_pred.append(obs_zeros)
            users_full.append(user)
     obs_for_pred = np.asarray(obs_for_pred)
     users_full = np.asarray(users_full)     
     return obs_for_pred ,users_full 
import wandb

import numpy as np
import pandas as pd
import math

import d3rlpy
from d3rlpy.base import LearnableBase
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics import evaluate_on_environment
from d3rlpy.algos import DiscreteCQL, DiscreteSAC#, SDAC

import rs_datasets
import argparse
import sys
sys.path.append(".")
from data_preparing.prepare_data import reidification, _prepare_data, item_user_pair
from rating_metrics.metrics import true_ndcg
from mdp.rating_mdp import original_actions, binary_actions,\
                                    negative_reward_binary_actions, negative_reward,mono_reward
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', help='int(k)', type = int)
    parser.add_argument('--emb', help='bool(random/als/ddpg)', type = str)
    parser.add_argument('--data_use', help='int(count_of_data)', type = int)
    parser.add_argument('--epochs', help='int(epochs)', type = int)
    parser.add_argument('--pfunc', help='str(run_name)', type = str)
    parser.add_argument('--wandb_run_name', help='str(run_name)', type = str)
    args = parser.parse_args()
    wandb.init(project="Rat MDP", group="CQL_als_emb", name = args.wandb_run_name)
    
    ml = pd.read_csv("ml_prepared.csv")
    raitings = ml[:args.data_use]
    #raitings = reidification(raitings)
    print(raitings)
    if args.pfunc == 'o':
        pfunc = original_actions
    elif args.pfunc == 'bina':        
        pfunc = binary_actions
    elif args.pfunc == 'nrba':        
        pfunc = negative_reward_binary_actions
    elif args.pfunc == 'nr':        
        pfunc = negative_reward
    elif args.pfunc == 'mr':        
        pfunc = mono_reward
    mdp, test_mdp, mapping_items, mapping_users, inv_mapp_items, inv_mapp_users, mask_test = _prepare_data(raitings, emb = args.emb, pfunc= pfunc)
    
   # print(len(mask_test), len(ml))
   # mask_train = raitings['dataset']=='train'
    items_obs_orig = np.unique(raitings[mask_test]['item_id'].values)
    users_obs_orig = np.unique(raitings[mask_test]['user_id'].values)
    
    items_obs = [mapping_items[item] for item in items_obs_orig]
    users_obs = [mapping_users[user] for user in users_obs_orig]
    
    obs_for_pred, users = item_user_pair(items_obs, users_obs)    
    scorer = true_ndcg(obs_for_pred, users, inv_mapp_items, top_k = args.top_k)
    
    algo = DiscreteCQL(use_gpu=False, batch_size=1024, n_critics = 1)
  #  print(args.epochs)
    algo.fit(mdp, eval_episodes=test_mdp,n_epochs = args.epochs, scorers={'NDCG': scorer})
    
    
    
import implicit
from scipy.sparse import coo_matrix

import numpy as np
import pandas as pd

from replay.models.rl.rating_mdp.embeddings.ddpg_embeddings import load_embeddings

    

    
def als_embeddings(df_full, emb_size=8):
    users = df_full['user_id']
    items = df_full['item_id']
    values = df_full['rating']
    sparse_matrix = coo_matrix((values, (users, items)))
    model = implicit.als.AlternatingLeastSquares(factors=8)
    model.fit(sparse_matrix)
    user_embeddings = model.user_factors
    item_embeddings = model.item_factors
    
    user_list = list(set(users))
    item_list = list(set(items))
    
    user_embeddings = [tuple(emb.tolist()) for emb in user_embeddings.to_numpy()]
    item_embeddings = [tuple(emb.tolist()) for emb in item_embeddings.to_numpy()]
                     
    user_mapping = dict(zip(user_list, user_embeddings))
    item_mapping = dict(zip(item_list, item_embeddings))   
                     
    user_inv_mapping = dict(zip(user_embeddings, user_list))
    item_inv_mapping = dict(zip(item_embeddings, item_list))
                     
    return user_mapping, user_inv_mapping, item_mapping, item_inv_mapping

def ddpg_embeddings(df_full):
    users = df_full['user_id']
    items = df_full['item_id']
    values = df_full['rating']
    user_embeddings, item_embeddings = load_embeddings(path_to_model = "model_final.pt", 
                                          model_params=[943, 1682, 8, 16, 5])
    
    user_list = list(set(users))
    item_list = list(set(items))
    
    user_embeddings = [tuple(emb.tolist()) for emb in user_embeddings.numpy()]
    item_embeddings = [tuple(emb.tolist()) for emb in item_embeddings.numpy()]
                     
    user_mapping = dict(zip(user_list, user_embeddings))
    item_mapping = dict(zip(item_list, item_embeddings))   
                     
    user_inv_mapping = dict(zip(user_embeddings, user_list))
    item_inv_mapping = dict(zip(item_embeddings, item_list))
                     
    return user_mapping, user_inv_mapping, item_mapping, item_inv_mapping
     
    
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
import os
import pickle
import numpy as np
import glob
from collections import Counter


def mapping_via_most_common(items, count_to_use = 10, group_range = None):
    groups = Counter(items).most_common()
    if group_range is None:
        items_to_use = groups[:count_to_use]
    else:
        items_to_use = [item for item in groups if group_range[0]<=item[1]<=group_range[1]]
        np.random.shuffle(items_to_use)
        items_to_use = items_to_use[:count_to_use]
       # print(items_to_use)
    sorted_items = [item[0] for item in items_to_use]
    max_v = items_to_use[0][1]
    mapping = dict(zip(sorted_items, list(range(len(sorted_items)))))
    return mapping, max_v


def trajectory4user(user_data, item_mapping, use_onehot = True, f_obs_modfier = None):
    #print(user_data)    
    observations = []
    rewards = []
    actions = []
    termaits = []
    vector_lenghts = len(item_mapping)
    for i in range(len(user_data)-1):
        #print(user_data['item_id'].values[:i+1])
        obs_values = list(map(lambda item: item_mapping[item], user_data['item_idx'].values[:i+1]))    
        if use_onehot:
            observation = np.zeros(vector_lenghts)
            observation[obs_values] = 1
        else:
            observation = obs_values
        action = user_data['item_idx'].values[i+1]
        user_action = user_data['event'].values[i+1]
        if user_action == 'view':
            reward = 0.5
        elif user_action == 'addtocart':
            reward = 1
        elif user_action == 'transaction':
            reward = 1.5
        
        observations.append(observation.tolist())
        actions.append(action.tolist())
        rewards.append(reward)
        termaits.append(0)
    termaits[-1] = 1
    return observations, actions, rewards, termaits    
        
def df2trajectories(data, item_mapping, use_onehot = True):
    observations = []
    actions = []
    rewards = []
    termaits = []
    
    users = list(set(data['user_idx']))
    items_count = data['item_idx'].max()
    min_item_vaue = data['item_idx'].min()

    for user_id in users:
        user_information = data[data['user_idx'] == user_id]
        u_observations,u_actions,u_rewards,u_termaits =\
                trajectory4user(data[data['user_idx'] == user_id], item_mapping = item_mapping,
                                use_onehot = use_onehot)
        observations += u_observations
        actions += u_actions
        rewards += u_rewards
        termaits += u_termaits
    return observations, actions, rewards, termaits



class RLDataPreparator():    
    def __init__(self, data = None, load_from_file = "data1000_GR_5_10", dataset_name = "data", onehot = True):
        self.data = data
        self.trajectories =None if not load_from_file else self.load(load_from_file)
        self.dataset_name = dataset_name
        self.onehot = True
        pass
    
    def prepare_data(self, count_to_use = 1000, group_range = (5,10)):
        if self.trajectories is None
        ### filter users
            users = list(self.data['user_idx'])
            user_filter, max_v = mapping_via_most_common(users,count_to_use = count_to_use, group_range = group_range )
            best_users = list(user_filter.keys())
            active_filter = self.data['user_idx'].isin(best_users)
            active_data = self.data[active_filter]

           # print(active_data)
            ### filter items
            active_items = list(active_data['item_idx'])
            item_mapping,_ = mapping_via_most_common(active_items, -1)        
            observations, actions, rewards, termaits = df2trajectories(active_data, item_mapping,
                                                                       use_onehot = self.onehot)
            self.trajectories = observations, actions, rewards, termaits        
            self.save(self.trajectories, self.dataset_name+f"{count_to_use}_GR_{group_range[0]}_{group_range[1]}")
        else:
            return self.trajectories

        return observations, actions, rewards, termaits
    
    def save(self, trajectories, file_name):
        os.makedirs("./data", exist_ok=True)
        with open(f'./data/{file_name}.pickle', 'wb') as f:
            pickle.dump(trajectories, f)
        print(f"Saved at ./data/{file_name}.pickle")
        return
    
    def load(self, file_name):
        with open(f'./data/{file_name}.pickle', 'rb') as f:
            trajectories = pickle.load(f)
        return trajectories
    
    
if __name__ == "__main__":
    from rs_datasets import RetailRocket
    data = RetailRocket()
    preparator_retail = RLDataPreparator(data.log.sort_values(['user_id','ts']))
    obs, _, _, _ = preparator_retail.prepare_data(count_to_use = 1000)
    
    preparator_retail = RLDataPreparator(load_from_file = 'data50_GR_5_10')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import gym
import numpy as np

import collections
import pickle

import d4rl
import d4rl.gym_mujoco
import os

DATASET_DIR = "" # your_pkl_pat

datasets = []
names = [
    "halfcheetah-medium-v2",
    "walker2d-medium-v2",
    "hopper-medium-v2",
    "halfcheetah-medium-replay-v2",
    "walker2d-medium-replay-v2",
    "hopper-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "walker2d-medium-expert-v2",
    "hopper-medium-expert-v2"
]


for env_name in names:
		name = env_name
		env = gym.make(name)
		print(env.observation_space)

		with open(f'{DATASET_DIR}/{env_name}_offline_pref.pkl', 'rb') as f:
			dataset = pickle.load(f)
		print(dataset.keys())

		if "rewards_ensembles" in dataset:
			new_rs_mean = dataset["rewards_ensembles"].mean(axis=1, keepdims=True)
			new_rs_std = dataset["rewards_ensembles"].std(axis=1, keepdims=True)
			new_rs = (dataset["rewards_ensembles"] - new_rs_mean) / new_rs_std

			dataset["rewards_ensembles"] = new_rs.mean(axis=0, keepdims=False)


		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = False
		if 'timeouts' in dataset:
			use_timeouts = True

		episode_step = 0
		paths = []
		for i in range(N):
			if "antmaze" in env_name:
				done_bool = False
			elif "terminals" in dataset:
				done_bool = bool(dataset['terminals'][i])
			else:
				done_bool = bool(dataset['dones'][i])
			
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = (episode_step == 1000-1)
			for k in ['observations', 'actions', 'rewards', 'terminals',"rewards_ensembles"]:
				k_ = k
				if "terminals" not in dataset and k_ is 'terminals':
					k_ = 'dones'
				data_[k].append(dataset[k_][i])

			episode_step += 1

			if final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			

		returns = np.array([np.sum(p['rewards']) for p in paths])
		
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(f'Number of samples collected: {num_samples}')
		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')


		with open(f'{name}.pkl', 'wb') as f:
			pickle.dump(paths, f)

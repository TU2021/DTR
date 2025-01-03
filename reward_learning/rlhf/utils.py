import collections
import numpy as np
import gym
from tqdm import trange
import torch
import torch.nn as nn
import math


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

def to_torch(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)

class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-5 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


@torch.no_grad()
def reward_from_preference(
    dataset: D4RLDataset,
    reward_model,
    batch_size: int = 256,
    reward_model_type: str = "transformer",
    device="cuda"
):
    # id_index = pref_dataset['index'].reshape(-1)
    data_size = dataset["rewards"].shape[0]
    interval = int(data_size / batch_size) + 1
    new_r = np.zeros_like(dataset["rewards"],dtype=float)
    new_rs = np.zeros((len(reward_model.ensemble),dataset["rewards"].shape[0]),dtype=float)


    if reward_model_type == "transformer":
        max_seq_len = reward_model.max_seq_len
        for each in reward_model.ensemble:
            each.eval()
 
        obs, act = [], []
        ptr = 0
        for i in trange(data_size):
            
            if len(obs) < max_seq_len:
                obs.append(dataset["observations"][i])
                act.append(dataset["actions"][i])
            
            if dataset["terminals"][i] > 0 or i == data_size - 1 or len(obs) == max_seq_len:
                tensor_obs = to_torch(np.array(obs)[None,], dtype=torch.float32).to(device)
                tensor_act = to_torch(np.array(act)[None,], dtype=torch.float32).to(device)
                
                new_reward = 0
                for each in reward_model.ensemble:
                    new_reward += each(tensor_obs, tensor_act).detach().cpu().numpy()
                new_reward /= len(reward_model.ensemble)
                if tensor_obs.shape[1] <= -1:
                    new_r[ptr:ptr+tensor_obs.shape[1]] = dataset["rewards"][ptr:ptr+tensor_obs.shape[1]]
                else:
                    new_r[ptr:ptr+tensor_obs.shape[1]] = new_reward
                ptr += tensor_obs.shape[1]
                obs, act = [], []
    else:
        for i in trange(interval):
            start_pt = i * batch_size
            end_pt = (i + 1) * batch_size

            observations = dataset["observations"][start_pt:end_pt]
            actions = dataset["actions"][start_pt:end_pt]
            obs_act = np.concatenate([observations, actions], axis=-1)

            new_reward, new_rewards = reward_model.get_reward_batch_ensemble(obs_act)

            new_r[start_pt:end_pt] = new_reward.reshape(-1)
            new_rs[:,start_pt:end_pt] = new_rewards.squeeze(-1)

    dataset["rewards"] = new_r
    dataset["rewards_ensembles"] = new_rs

    return dataset

@torch.no_grad()
def pref_dataset_reward_from_preference(
    dataset: D4RLDataset,
    reward_model,
    batch_size: int = 256,
    reward_model_type: str = "transformer",
    device="cuda"
):
    data_size = dataset["terminals"].shape[0]
    # interval = int(data_size / batch_size) + 1
    interval = data_size
    new_r = np.zeros_like(dataset["terminals"],dtype=float)
    new_r_std = np.zeros_like(dataset["terminals"],dtype=float)
    new_r_min = np.zeros_like(dataset["terminals"],dtype=float)
    
    if "transformer" in reward_model_type:
        max_seq_len = reward_model.max_seq_len
        for each in reward_model.ensemble:
            each.eval()
 
        obs, act, reward = [], [], []
        ptr = 0
        for i in trange(interval):
            
            obs=dataset["observations"][i]
            act=dataset["actions"][i]
            tensor_obs = to_torch(np.array(obs)[None,], dtype=torch.float32).to(device)
            tensor_act = to_torch(np.array(act)[None,], dtype=torch.float32).to(device)
            new_reward,weight,score = 0,0,0
            for each in reward_model.ensemble:
                new_reward_ = each(tensor_obs, tensor_act)
                new_reward = new_reward + new_reward_.detach().cpu().numpy()
            new_reward /= len(reward_model.ensemble)
            if tensor_obs.shape[1] <= -1:
                new_r[ptr:ptr+tensor_obs.shape[1]] = dataset["rewards"][ptr:ptr+tensor_obs.shape[1]]
            else:
                new_r[ptr:ptr+tensor_obs.shape[1]] = new_reward
            ptr += tensor_obs.shape[1]
            obs, act = [], []
    else:
        for i in trange(interval):
            observations = dataset["observations"][i]
            actions = dataset["actions"][i]
            obs_act = np.concatenate([observations, actions], axis=-1)

            new_reward, new_reward_std, new_reward_min = reward_model.get_reward_batch(obs_act)
            new_r[i,:] = new_reward.reshape(-1)
            new_r_std[i,:] = new_reward_std.reshape(-1)
            new_r_min[i,:] = new_reward_min.reshape(-1)
    
    dataset["rewards"] = new_r.copy()
    dataset["rewards_std"] = new_r_std.copy()
    dataset["rewards_min"] = new_r_min.copy()
    # return dataset
    new_dataset = []
    for i in range(interval):
        entry = {
            'observations': dataset['observations'][i],
            'actions': dataset['actions'][i],
            'rewards': dataset['rewards'][i],
            'rewards_std': dataset['rewards_std'][i],
            'rewards_min': dataset['rewards_min'][i],
            'timesteps': dataset['timesteps'][i],
            'terminals': dataset['terminals'][i].astype(bool)  # 将0/1转换为True/False
        }
        new_dataset.append(entry)
    return new_dataset


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PrefTransformer1(nn.Module):
    ''' Transformer Structure used in Preference Transformer.
    
    Description:
        This structure holds a causal transformer, which takes in a sequence of observations and actions, 
        and outputs a sequence of latent vectors. Then, pass the latent vectors through self-attention to
        get a weight vector, which is used to weight the latent vectors to get the final preference score.
    
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - max_seq_len: maximum length of sequence
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        max_seq_len: int = 100,
        d_model: int = 256, nhead: int = 4, num_layers: int = 1, 
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.causual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers
        )
        self.mask = nn.Transformer.generate_square_subsequent_mask(2*self.max_seq_len)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.r_proj = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        if self.mask.device != obs.device: self.mask = self.mask.to(obs.device)
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.causual_transformer(x, self.mask[:2*traj_len,:2*traj_len])[:, 1::2]
        # x: (batch_size, traj_len, d_model)

        q = self.q_proj(x) # (batch_size, traj_len, d_model)
        k = self.k_proj(x) # (batch_size, traj_len, d_model)
        r = self.r_proj(x) # (batch_size, traj_len, 1)
        
        w = torch.softmax(q@k.permute(0, 2, 1)/np.sqrt(self.d_model), -1).mean(-2)
        # w: (batch_size, traj_len)
        
        z = (w * r.squeeze(-1)) # (batch_size, traj_len)
        
        return torch.tanh(z)
        # return z


class PrefTransformer2(nn.Module):
    ''' Preference Transformer with no causal mask and no self-attention but one transformer layer to get the weight vector.
    
    Description:
        This structure has no causal mask and no self-attention.
        Instead, it uses one transformer layer to get the weight vector.
        
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        d_model: int, nhead: int, num_layers: int, 
    ):
        super().__init__()
        while num_layers < 2: num_layers += 1
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers - 1
        )
        self.value_layer = nn.Sequential(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 1
        ), nn.Linear(d_model, 1))
        self.weight_layer = nn.Sequential(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 1
        ), nn.Linear(d_model, 1))

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act
        
        x = self.transformer(x)[:, 1::2]
        v = self.value_layer(x)
        w = torch.softmax(self.weight_layer(x), 1)
        return (w*v).squeeze(-1)
    

class PrefTransformer3(nn.Module):
    ''' Preference Transformer with no causal mask and no weight vector.
    
    Description:
        This structure has no causal mask and even no weight vector.
        Instead, it directly outputs the preference score.
        
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        d_model: int, nhead: int, num_layers: int, 
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers
        )
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act
        
        x = self.transformer(x)[:, 1::2]
        return self.output_layer(x).squeeze(-1)
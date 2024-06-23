import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from tqdm import tqdm
import wandb
import os
import gym
from train_ppo import train_ppo
import pickle
from env.wei_obs_grid import WeightedObsGrid
from utils.utils import uniform_weight, normal_weight, NormalWeightGrid

def create_new_env():
    grid_size = (10, 10)
    return WeightedObsGrid(
        grid_size=grid_size,
        start=(0, 9),
        goal=(3, 2),
        obstacles=[(4,4), (1,1), (2,2), (3,3)],
        weights=np.random.rand(grid_size[0], grid_size[1]),
        dis_reward=True,
        wei_reset=NormalWeightGrid(grid_size),
        goal_set=[(3, 2), (3, 6), (6, 2),(9, 9)]
    )


# 参数设置

env = create_new_env()
env = pickle.load(open('env_end_rand.pkl', 'rb'))
env.dis_reward = True
env.render()
actor_lr = 1e-5
critic_lr = 5e-4
num_episodes = 5000
hidden_dim = 3
gamma = 0.9
epsilon = 0.2
lmbda = 0.9
epochs = 10
reset_interval = 5000
max_steps = 500
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

train_ppo(env, actor_lr, critic_lr, num_episodes, hidden_dim, gamma, epsilon, lmbda, epochs, reset_interval, max_steps)

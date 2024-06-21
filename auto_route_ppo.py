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

def create_new_env(grid_size, start, goal, obstacles, weights):
    return WeightedObsGrid(
        grid_size=grid_size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        weights=weights, 
        wei_reset=NormalWeightGrid(grid_size)
    )


# 参数设置
# grid_size = (50, 50)
# start = (0, 0)
# goal = (49, 49)
# obstacles = random.sample([(i, j) for i in range(50) for j in range(50) if (i, j) != start and (i, j) != goal], 100)
# weights = np.random.rand(grid_size[0], grid_size[1])
# env = create_new_env(grid_size, start, goal, obstacles, weights)
env = pickle.load(open('env.pkl', 'rb'))
env.render()
actor_lr = 1e-4
critic_lr = 5e-3
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
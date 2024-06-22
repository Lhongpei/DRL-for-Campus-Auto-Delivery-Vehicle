import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from tqdm import tqdm
import wandb
import os
import gym
from train_ppo import train_ppo
from models.ppo import PPO
from env.wei_obs_grid import WeightedObsGrid
from utils.utils import uniform_weight, normal_weight, NormalWeightGrid
import pickle
from baseline.dijkstra import dijkstra_grid_with_weights, reconstruct_path
import copy
def create_new_env(grid_size, start, goal, obstacles, weights):
    return WeightedObsGrid(
        grid_size=grid_size,
        start=start,
        goal=goal,
        obstacles=obstacles,
        weights=weights, 
        wei_reset=NormalWeightGrid(grid_size)
    )
    
grid_size = (50, 50)
start = (0, 0)
goal = (49, 49)
obstacles = random.sample([(i, j) for i in range(50) for j in range(50) if (i, j) != start and (i, j) != goal], 100)
weights = np.random.rand(grid_size[0], grid_size[1])
env = WeightedObsGrid(grid_size, start, goal, obstacles, weights, NormalWeightGrid(grid_size))
pickle.dump(env, open('env.pkl', 'wb'))
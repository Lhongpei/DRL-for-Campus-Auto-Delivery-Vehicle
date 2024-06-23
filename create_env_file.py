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
    
grid_size = (20, 20)
start = (0, 0)
goal = (19, 19)
length_end_set = 5
obstacles_num = 10
end_set = []
obstacles = random.sample([(i, j) for i in range(grid_size[0]) for j in range(grid_size[1]) if (i, j) != start and (i, j) != goal], obstacles_num)
weights = np.random.rand(grid_size[0], grid_size[1])
for i in range(length_end_set):
    while True:
        end = (random.randint(0, grid_size[0]-1), random.randint(0, grid_size[1]-1))
        if end not in obstacles and end != start:
            end_set.append(end)
            break
env = WeightedObsGrid(grid_size, start, goal, obstacles, weights, NormalWeightGrid(grid_size), goal_set=end_set)
pickle.dump(env, open('env_end_rand.pkl', 'wb'))
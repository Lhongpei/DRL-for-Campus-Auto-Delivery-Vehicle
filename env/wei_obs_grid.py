import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from utils.utils import select_k_coordinates
class WeightedObsGrid(gym.Env):
    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=None, weights=None, wei_reset=None, dis_reward = True):
        super(WeightedObsGrid, self).__init__()
        self.weights = weights if weights is not None else {}
        self.start = start
        self.goal = goal
        self.max_distance = np.linalg.norm(np.array(grid_size) - np.array([1, 1]))
        self.action_space = spaces.Discrete(4)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.grid_size = grid_size
        self.total_cost = 0
        self.total_steps = 0
        self.obstacles = obstacles
        self.wei_reset = wei_reset
        self.dis_reward = dis_reward
        self.observation_space = spaces.Dict({
            'current_position': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            'end_position': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            'weight': spaces.Box(low=0, high=1, shape=grid_size, dtype=np.float32),
            'obstacles': spaces.Box(low=0, high=1, shape=grid_size, dtype=np.float32),
            'history': spaces.Box(low=0, high=1, shape=grid_size, dtype=np.float32)
        })
        self.state = {
            'current_position': np.array(start, dtype=np.float32),
            'end_position': np.array(goal, dtype=np.float32),
            'weight': np.ones(grid_size, dtype=np.float32),
            'obstacles': np.zeros(grid_size, dtype=np.float32),
            'history': np.zeros(grid_size, dtype=np.float32)
        }
        self.state['history'][start] = 1
        if obstacles:
            for obs in obstacles:
                self.state['obstacles'][obs] = 1
        
        if weights is not None:
            self.state['weight'] = weights
                
    def reset(self, 
              seed=None, 
              start_random=False, 
              end_random=False, 
              wei_reset=None # Function to reset weights
              ):
        self.total_cost = 0
        self.total_steps = 0
        self.state['current_position'] = np.array(self.start, dtype=np.float32)
        self.state['history'] = np.zeros(self.grid_size, dtype=np.float32)
        if seed:
            np.random.seed(seed)
        if start_random and end_random:
            select_s = select_k_coordinates(self.grid_size, 2, self.obstacles)
            self.state['current_position'] = np.array(select_s[0], dtype=np.float32)
            self.state['history'][select_s[0]] = 1
            self.goal = select_s[1]
            self.start = select_s[0]
            self.state['end_position'] = np.array(select_s[1], dtype=np.float32)
        elif start_random:
            select_s = select_k_coordinates(self.grid_size, 1, self.obstacles)
            while np.array_equal(select_s[0], self.goal):
                select_s = select_k_coordinates(self.grid_size, 1, self.obstacles)
            self.state['current_position'] = np.array(select_s[0], dtype=np.float32)
            self.state['history'][select_s[0]] = 1
            self.start = select_s[0]
        elif end_random:
            select_s = select_k_coordinates(self.grid_size, 1, self.obstacles)
            while np.array_equal(select_s[0], self.start):
                select_s = select_k_coordinates(self.grid_size, 1, self.obstacles)
            self.goal = select_s[0]
            self.state['end_position'] = np.array(select_s[0], dtype=np.float32)
            
        if wei_reset is not None:
            self.state['weight'] = wei_reset(self.grid_size)
        
        return self.processed_state()
    
    def eligible_action_idxes(self):
        el_indicate = [0, 0, 0, 0]
        for i, direction in enumerate(self.directions):
            new_position = (
                int(self.state['current_position'][0] + direction[0]),
                int(self.state['current_position'][1] + direction[1])
            )
            if (
                0 <= new_position[0] < self.observation_space['weight'].shape[0] and
                0 <= new_position[1] < self.observation_space['weight'].shape[1] and
                not self.state['obstacles'][new_position]
                ):
                el_indicate[i] = 1
        return np.array(el_indicate)
    
    def step(self, action):
        new_position = (
            int(self.state['current_position'][0] + self.directions[action][0]),
            int(self.state['current_position'][1] + self.directions[action][1])
        )
        
        if (0 <= new_position[0] < self.observation_space['weight'].shape[0] and
            0 <= new_position[1] < self.observation_space['weight'].shape[1] and
            not self.state['obstacles'][new_position]):
            
            old_distance = np.linalg.norm(self.state['current_position'] - self.state['end_position'])
            self.state['current_position'] = np.array(new_position, dtype=np.float32)
            self.state['history'][new_position] += 1
            
            if self.dis_reward:
                new_distance = np.linalg.norm(self.state['current_position'] - self.state['end_position'])
                distance_change = old_distance - new_distance
                if distance_change == 0:
                    reward = -1/2
                elif distance_change > 0:
                    reward = 0
                else:
                    reward = -1/2 + distance_change
            else:
                reward = 0
            #reward = distance_change   # 正奖励，距离减少越多，奖励越大
        else:
            raise ValueError('Invalid action')
        
        if np.array_equal(self.state['current_position'], self.goal):
            reward += 2*max(self.grid_size[0], self.grid_size[1])  # 到达终点给予正奖励
            done = True
        else:
            reward += -1 * self.state['weight'][new_position[0], new_position[1]]
            done = False
        self.total_cost += self.state['weight'][new_position[0], new_position[1]]
        self.total_steps += 1
        if self.wei_reset is not None:
            self.state['weight'] = self.wei_reset.generate_weights()
        return self.processed_state(), reward, done, {}
    
    def processed_state(self):
        current_position = np.zeros(self.grid_size, dtype=np.float32)
        current_position[tuple(self.state['current_position'].astype(int))] = 1
        end_position = np.zeros(self.grid_size, dtype=np.float32)
        end_position[tuple(self.state['end_position'].astype(int))] = 1
        # hist_onehot = np.array(self.state['history'] > 0, dtype=np.float32)
        return np.stack([current_position, end_position, self.state['weight'], self.state['obstacles']], axis=0)
    
    def render(self, exact_rate = False):
        grid_size = self.state['weight'].shape
        fig, ax = plt.subplots()
        
        ax.set_xticks(np.arange(grid_size[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(grid_size[0]) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        
        background = np.zeros(grid_size)
        background[self.state['obstacles'] == 1] = -1
        background[self.state['history'] == 1] = -2
        
        cmap = plt.cm.get_cmap('gray', 3) 
        ax.imshow(background, cmap=cmap, interpolation='nearest', extent=(-0.5, grid_size[1]-0.5, grid_size[0]-0.5, -0.5))
        
        if exact_rate:
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    if self.state['obstacles'][i, j] == 1: 
                        continue
                    ax.text(j, i, f"{self.state['weight'][i, j]:.2f}", ha='center', va='center', color='black')
        
        
        
        start = self.start
        goal = self.goal
        ax.add_patch(patches.Circle((start[1], start[0]), 0.4, edgecolor='black', facecolor='green'))
        ax.add_patch(patches.Circle((goal[1], goal[0]), 0.4, edgecolor='black', facecolor='red'))
        
        current_position = self.state['current_position']
        ax.add_patch(patches.Circle((current_position[1], current_position[0]), 0.4, edgecolor='black', facecolor='yellow'))
        
        plt.gca().invert_yaxis()
        plt.show()

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def preprocess_state(state):
    current_position = np.zeros(state['weight'].shape, dtype=np.float32)
    current_position[tuple(state['current_position'].astype(int))] = 1
    return np.stack([current_position, state['weight'], state['obstacles'], state['history']], axis=0)


class WeightedObsGrid(gym.Env):
    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=None, weights=None):
        super(WeightedObsGrid, self).__init__()
        self.weights = weights if weights is not None else {}
        self.start = start
        self.goal = goal
        
        self.action_space = spaces.Discrete(4)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        self.observation_space = spaces.Dict({
            'current_position': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            'weight': spaces.Box(low=0, high=1, shape=grid_size, dtype=np.float32),
            'obstacles': spaces.Box(low=0, high=1, shape=grid_size, dtype=np.float32),
            'history': spaces.Box(low=0, high=1, shape=grid_size, dtype=np.float32)
        })
        self.state = {
            'current_position': np.array([0, 0], dtype=np.float32),
            'weight': np.ones(grid_size, dtype=np.float32),
            'obstacles': np.zeros(grid_size, dtype=np.float32),
            'history': np.zeros(grid_size, dtype=np.float32)
        }
        if obstacles:
            for obs in obstacles:
                self.state['obstacles'][obs] = 1
        
        if weights is not None:
            self.state['weight'] = weights
                
    def reset(self, seed=None):
        self.state['current_position'] = np.array(self.start, dtype=np.float32)
        self.state['history'] = np.zeros(self.observation_space['history'].n, dtype=np.float32)
        if seed:
            np.random.seed(seed)
        return self.state
    
    def step(self, action):
        new_position = (
            self.state['current_position'][0] + self.directions[action][0],
            self.state['current_position'][1] + self.directions[action][1]
        )
        
        if (0 <= new_position[0] < self.observation_space['weight'].shape[0] and
            0 <= new_position[1] < self.observation_space['weight'].shape[1] and
            not self.state['obstacles'][new_position]):
            self.state['current_position'] = np.array(new_position, dtype=np.float32)
            self.state['history'][new_position] = 1
            
        if np.array_equal(self.state['current_position'], self.goal):
            reward = 100
            done = True
        else:
            reward = -1*self.state['weight'][new_position]
            done = False
            
        return self.state, reward, done, {}
    
    def render(self):
        grid_size = self.state['weight'].shape
        fig, ax = plt.subplots()
        
        ax.set_xticks(np.arange(grid_size[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(grid_size[0]) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        
        # 绘制背景
        background = np.zeros(grid_size)
        background[self.state['obstacles'] == 1] = -1
        background[self.state['history'] == 1] = -2
        
        cmap = plt.cm.get_cmap('gray', 3)  # 自定义颜色映射
        ax.imshow(background, cmap=cmap, interpolation='nearest', extent=(-0.5, grid_size[1]-0.5, grid_size[0]-0.5, -0.5))
        
        # 绘制权重值
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                ax.text(j, i, f"{self.state['weight'][i, j]:.2f}", ha='center', va='center', color='black')
        
        # 绘制起点和终点
        start = self.start
        goal = self.goal
        ax.add_patch(patches.Circle((start[1], start[0]), 0.4, edgecolor='black', facecolor='green'))
        ax.add_patch(patches.Circle((goal[1], goal[0]), 0.4, edgecolor='black', facecolor='red'))
        
        # 绘制当前位置
        current_position = self.state['current_position']
        ax.add_patch(patches.Circle((current_position[1], current_position[0]), 0.4, edgecolor='black', facecolor='yellow'))
        
        plt.gca().invert_yaxis()
        plt.show()

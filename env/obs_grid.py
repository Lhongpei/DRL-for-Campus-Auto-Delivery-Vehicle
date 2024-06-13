import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class ObsGrid(gym.Env):
    def __init__(self, grid_size=(10, 10), start=(0, 0), goal=(9, 9), obstacles=None):
        super(ObsGrid, self).__init__()
        
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = start
        self.obstacles = obstacles if obstacles else []
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.int32)
        
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.grid = np.zeros(self.grid_size) 
        for obs in self.obstacles:
            self.grid[obs] = 1
        
        
    def reset(self):
        self.grid = np.zeros(self.grid_size)
        for obs in self.obstacles:
            self.grid[obs] = 1
        self.state = self.start
        return np.array(self.state, dtype=np.int32)
    
    def step(self, action):
        new_state = (
            self.state[0] + self.directions[action][0],
            self.state[1] + self.directions[action][1]
        )
        
        if (0 <= new_state[0] < self.grid_size[0] and
            0 <= new_state[1] < self.grid_size[1] and
            new_state not in self.obstacles):
            self.state = new_state
            self.grid[new_state] += 0.2
        
        if self.state == self.goal:
            reward = 100
            done = True

        else:
            reward = -1
            done = False
        
        return np.array(self.state, dtype=np.int32), reward, done, {}
    
    def render(self): 
        
        plt.imshow(self.grid, cmap='gray')
        plt.show()

if __name__ == "__main__":
# 测试环境
    env = ObsGrid(
        grid_size=(10, 10),
        start=(0, 0),
        goal=(9, 9),
        obstacles=[(1, 1), (2, 2), (3, 3), (4, 4)]
    )

    state = env.reset()

    for _ in range(20):
        action = env.action_space.sample()  # 随机选择一个动作
        state, reward, done, _ = env.step(action)
        
        if done:
            print(f"Goal reached with reward: {reward}")
            break
    env.render()
    env.close()

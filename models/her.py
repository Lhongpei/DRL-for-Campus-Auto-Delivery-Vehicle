import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.nn import CNNEmb, Actor, Critic
import copy
class Trajectory:

    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1


class ReplayBuffer_Trajectory:

    def __init__(self, capacity, goal_reward):
        self.buffer = collections.deque(maxlen=capacity)
        self.goal_reward = goal_reward

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, use_her, her_ratio=0.8):
        batch = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer, 1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state]
            next_state = traj.states[step_state + 1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.dones[step_state]

            if use_her and np.random.uniform() <= her_ratio:
                step_goal = np.random.randint(step_state + 1, traj.length + 1)
                goal_pos = np.array(np.nonzero(traj.states[step_goal][0])).transpose()[0]  # 使用HER算法的future方案设置目标
                cur_pos = np.array(np.nonzero(traj.states[step_state][0])).transpose()[0]
                if not np.equal(goal_pos, cur_pos).all():
                    reward = -1 * traj.states[step_state][2][cur_pos[0], cur_pos[1]]  
                    done = False
                else:
                    reward = self.goal_reward
                    done = True
                state = copy.deepcopy(state)
                next_state = copy.deepcopy(next_state)
                
                state[1] = copy.deepcopy(traj.states[step_goal][0])
                next_state[1] = copy.deepcopy(traj.states[step_goal][0])
                

            batch['states'].append(state)
            batch['next_states'].append(next_state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

        batch['states'] = np.array(batch['states'])
        batch['next_states'] = np.array(batch['next_states'])
        batch['actions'] = np.array(batch['actions'])
        return batch
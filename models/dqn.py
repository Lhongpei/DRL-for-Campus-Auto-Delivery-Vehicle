import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.qnet import Qnet, DuelingCNNQnet
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma 
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    def max_next_q_values(self, next_states):
        return self.target_q_net(next_states).max(1)[0].view(-1, 1) 
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions) 
        # 下个状态的最大Q值
        max_next_q_values = self.max_next_q_values(next_states)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad() 
        dqn_loss.backward() 
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1
        
class Double_DQN(DQN):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        super().__init__(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)
        pass
    
    def max_next_q_values(self, next_states):
        q_values = self.q_net(next_states)
        actions = q_values.argmax(dim=1).view(-1, 1)
        return self.target_q_net(next_states).gather(1, actions)

class Dueling_DQN(DQN):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        super().__init__(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)
        self.q_net = DuelingCNNQnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        self.target_q_net = DuelingCNNQnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)    
    
class D3QN(Double_DQN):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        super().__init__(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)
        self.q_net = DuelingCNNQnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        self.target_q_net = DuelingCNNQnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)




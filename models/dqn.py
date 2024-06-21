import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.nn import Qnet, DuelingCNNQnet, DuelingQnet

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def add_traject(self, state_list, action_list, reward_list, next_state_list, done_list):
        for i in range(len(state_list)):
            self.add(state_list[i], action_list[i], reward_list[i], next_state_list[i], done_list[i])
    
    def size(self):
        return len(self.buffer)
    
class PrioritizedReplayBuffer:

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.alpha = alpha  # priority exponent

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities, dtype=np.float32)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
        else:
            probabilities = np.ones(len(self.buffer)) / len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)

class DQN(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_decay, target_update, device):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma 
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.count = 0
        self.device = device

    def epsilon_update(self):
        self.epsilon *= self.epsilon_decay
    
    def take_action(self, state, eligibles):
        if np.random.rand() < self.epsilon:
            eligibles_indices = np.where(eligibles==1)[0]
            action = np.random.choice(eligibles_indices)
            # action = np.random.choice(range(self.action_dim))
            
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
            prob = torch.zeros(self.action_dim, device=self.device)
            prob[eligibles==1] = F.softmax(self.q_net(state)[0][eligibles==1], dim=0)
            action = prob.argmax().item()
            # action = self.q_net(state).argmax().item()
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
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_decay, target_update, device):
        super().__init__(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_decay, target_update, device)
        pass
    
    def max_next_q_values(self, next_states):
        q_values = self.q_net(next_states)
        actions = q_values.argmax(dim=1).view(-1, 1)
        return self.target_q_net(next_states).gather(1, actions)

class Dueling_DQN(DQN):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_decay, target_update, device):
        super().__init__(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_decay, target_update, device)
        self.q_net = DuelingQnet(state_dim, hidden_dim,
                          self.action_dim).to(device)
        self.target_q_net = DuelingQnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)    
    
class D3QN(Double_DQN):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_decay, target_update, device):
        super().__init__(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, epsilon_decay, target_update, device)
        self.q_net = DuelingCNNQnet(state_dim, hidden_dim,
                          self.action_dim).to(device)
        self.target_q_net = DuelingCNNQnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        
    # def take_action(self, state, eligibles):
        
    #     state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
    #     prob = torch.zeros(self.action_dim, device=self.device)
    #     prob[eligibles] = F.softmax(self.q_net(state)[0][eligibles], dim=0)
    #     action = torch.multinomial(prob, 1).item()
    #     # action = self.q_net(state).argmax().item()
    #     return action




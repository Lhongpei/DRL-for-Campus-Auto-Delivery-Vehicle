import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.nn import CNNEmb, Actor, Critic
class PPO(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr , gamma,
                 epsilon, lmbda, epochs, device):
        super(PPO, self).__init__()
        self.action_dim = action_dim
        
        self.embedding = CNNEmb(state_dim, hidden_dim).to(device)
        self.actor_emb = Actor(self.embedding.conv_output_size, action_dim).to(device)
        self.critic_emb = Critic(self.embedding.conv_output_size).to(device)
        
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor_emb.parameters()) + list(self.embedding.parameters()),
            lr=actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_emb.parameters()) + list(self.embedding.parameters()),
            lr=critic_lr
        )
        
        self.gamma = gamma 
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.epochs = epochs

        self.count = 0
        self.device = device

    def actor(self, state, eligibles):
        emb = self.embedding(state)
        act = self.actor_emb(emb)
        eligibles = eligibles.reshape(-1, self.action_dim)
        masked_act = act.clone()
        masked_act[eligibles == 0] = -float('inf')
        prob = F.softmax(masked_act, dim=1)
        prob = torch.clamp(prob, min=1e-10)
        return prob
    
    def critic(self, state):
        emb = self.embedding(state)
        return self.critic_emb(emb)
    
    def take_action(self, state, eligibles):
        
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        prob = self.actor(state, eligibles)
        
        action_dist = torch.distributions.Categorical(prob)
        action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        
        return action.item()
    

    
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
        eligibles = torch.tensor(transition_dict['eligible'],
                                dtype=torch.float).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states, eligibles).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            #print('actor_prob', self.actor(states, eligibles))
            log_probs = torch.log(self.actor(states, eligibles).gather(1, actions))
            #print('log_probs', log_probs)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon,
                                1 + self.epsilon) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_np = np.array(advantage_list)
    return torch.tensor(advantage_np, dtype=torch.float)
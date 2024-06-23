import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.nn import CNNEmb, Policy, CriticQnet

class SAC(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, gamma,
                 target_entropy, tau, device):
        super(SAC, self).__init__()
        self.action_dim = action_dim

        self.actor_emb = Policy(state_dim, action_dim=4, hidden_dim=hidden_dim).to(device)
        self.q_net_1 = CriticQnet(state_dim, hidden_dim=hidden_dim).to(device)
        self.q_net_2 = CriticQnet(state_dim, hidden_dim=hidden_dim).to(device)
        self.q_target_net_1 = CriticQnet(state_dim, hidden_dim=hidden_dim).to(device)
        self.q_target_net_2 = CriticQnet(state_dim, hidden_dim=hidden_dim).to(device)

        self.q_target_net_1.load_state_dict(self.q_net_1.state_dict())
        self.q_target_net_2.load_state_dict(self.q_net_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor_emb.parameters(),
            lr=actor_lr
        )
        self.critic_1_optimizer = torch.optim.Adam(
            self.q_net_1.parameters(),
            lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.q_net_1.parameters(),
            lr=critic_lr
        )

        self.log_alpha = torch.tensor(np.log(0.01)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy
        self.tau = tau
        self.gamma = gamma
        self.count = 0
        self.device = device

    def actor(self, state, eligibles):
        act = self.actor_emb(state)
        eligibles = eligibles.reshape(-1, self.action_dim)
        masked_act = act.clone()
        masked_act[eligibles == 0] = -float('inf')
        prob = F.softmax(masked_act, dim=1)
        prob = torch.clamp(prob, min=1e-10)
        return prob

    def critic(self, state):
        
        return self.q_net_1(state), self.q_net_2(state)

    def target_critic(self, state):

        return self.q_target_net_1(state), self.q_target_net_2(state)

    def take_action(self, state, eligibles):
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        eligibles = torch.tensor(eligibles, dtype=torch.float).to(self.device).unsqueeze(0)
        probs = self.actor(state, eligibles)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states, torch.ones(next_states.size(0), self.action_dim).to(self.device))
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1, q2 = self.target_critic(next_states)
        min_q = torch.sum(next_probs * (torch.min(q1, q2) - self.log_alpha.exp() * entropy), dim=1, keepdim=True)
        next_value = min_q + self.log_alpha.exp() * entropy
        target = rewards + self.gamma * (1 - dones) * next_value
        return target

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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

        td_target = self.calc_target(rewards, next_states, dones)
        q1, q2 = self.critic(states)
        q1_values = q1.gather(1, actions)
        q2_values = q2.gather(1, actions)
        q1_loss = F.mse_loss(q1_values, td_target.detach())
        q2_loss = F.mse_loss(q2_values, td_target.detach())
        self.critic_1_optimizer.zero_grad()
        q1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        q2_loss.backward()
        self.critic_2_optimizer.step()

        probs = self.actor(next_states, torch.ones(next_states.size(0), self.action_dim).to(self.device))
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1, q2 = self.critic(states)

        q_values = torch.min(q1, q2)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - q_values)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.q_target_net_1, self.q_net_1)
        self.soft_update(self.q_target_net_2, self.q_net_2)

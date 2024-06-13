import torch
import torch.nn as nn
class Qnet(nn.Module):
    def __init__(self, action_dim, hidden_dim, input_shape):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingCNNQnet(nn.Module):
    def __init__(self, action_size, hidden_dim, input_shape):
        super(DuelingCNNQnet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        
        self.value_fc = nn.Linear(hidden_dim * 2 * input_shape[1] * input_shape[2], hidden_dim * 4)
        self.value = nn.Linear(hidden_dim * 4, 1)
        
        self.advantage_fc = nn.Linear(hidden_dim * 2 * input_shape[1] * input_shape[2], hidden_dim * 4)
        self.advantage = nn.Linear(hidden_dim * 4, action_size)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        
        value = torch.relu(self.value_fc(x))
        value = self.value(value)
        
        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
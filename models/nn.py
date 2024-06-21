import torch
import torch.nn as nn
import torch.nn.functional as F
class Qnet(nn.Module):
    def __init__(self, input_shape, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten() 
        
        conv_output_size = hidden_dim * 2 * input_shape[1] * input_shape[2]  # 确保这里计算正确
        self.fc1 = nn.Linear(conv_output_size, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, action_dim*4)
        self.fc4 = nn.Linear(action_dim*4, action_dim)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

class DuelingCNNQnet(nn.Module):
    def __init__(self, input_shape, hidden_dim, action_dim):
        super(DuelingCNNQnet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        
        conv_output_size = hidden_dim * 2 * input_shape[1] * input_shape[2]  # 确保这里计算正确
        self.value = nn.Sequential(
            nn.Linear(conv_output_size, 24),
            
            nn.ReLU(),
            nn.Linear(24, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(conv_output_size, action_dim * 2),
            nn.ReLU(),
            nn.Linear(action_dim *2, action_dim)
        )
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x)->torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.flatten(x)
        value = self.value(x)
        advantage = self.advantage(x)
        
        q_values = value + (advantage - advantage.mean(dim=1).view(-1,1))
        return q_values
    
class DuelingQnet(nn.Module):
    def __init__(self, input_shape, hidden_dim, action_dim):
        super(DuelingQnet, self).__init__()
        self.flatten = nn.Flatten()
        flatten_size = 4 * input_shape[1] * input_shape[2]  # 确保这里计算正确
        self.value = nn.Sequential(
            nn.Linear(flatten_size, 24),
            
            nn.ReLU(),
            nn.Linear(24, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(flatten_size, action_dim)
        )
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x)->torch.Tensor:
        
        x = self.flatten(x)
        value = self.value(x)
        advantage = self.advantage(x)
        
        q_values = value + (advantage - advantage.mean(dim=1).view(-1,1))
        return q_values
    
class CNNEmb(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(CNNEmb, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.init_weights()
        self.conv_output_size = hidden_dim * 2 * input_shape[1] * input_shape[2]
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x) ->torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        return x

class Actor(nn.Module):
    def __init__(self, in_channel, action_dim, hidden_dim = 32):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(in_channel, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            
    def forward(self, cnn_embedding):
        x = F.relu(self.fc1(cnn_embedding))
        x = self.fc2(x)
        return x
        
class Critic(nn.Module):
    def __init__(self, in_channel, hidden_dim = 32):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_channel, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
    def forward(self, cnn_embedding):
        x = F.relu(self.fc1(cnn_embedding))
        x = self.fc2(x)
        return x
                
        

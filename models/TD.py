from collections import defaultdict
import numpy as np
class Args:
    def __init__(self, lr, gamma, epsilon, train_episodes):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.train_episodes = train_episodes
        
class Sarsa:
    def __init__(self, n_actions, args):
        self.n_actions = n_actions  
        self.lr = args.lr  
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state):
        # please complete the code for choosing action
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.n_actions)
        return action
    
    def update(self, state, action, reward, next_state, next_action, done):

        if done:
            next_q_value_est = reward
        else:
            next_q_value_est = reward + self.gamma * self.Q_table[str(next_state)][next_action]
        self.Q_table[str(state)][action] += self.lr * (next_q_value_est - self.Q_table[str(state)][action])
        
    def get_opt_action(self, state):
        return np.argmax(self.Q_table[str(state)])
    

class QLearning:
    def __init__(self, n_actions, args):
        self.n_actions = n_actions 
        self.lr = args.lr
        self.gamma = args.gamma  
        self.sample_count = 0  
        self.epsilon = args.epsilon
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(self, state):
        # please complete the code for choosing action
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.n_actions)
        return action

    def update(self, state, action, reward, next_state, done):

        now_q_value = self.Q_table[str(state)][action]
        if done: 
            next_q_value_est = reward
        else:
            next_q_value_est = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (next_q_value_est - now_q_value)

    def get_opt_action(self, state):
        action = np.argmax(self.Q_table[str(state)])
        return action
    
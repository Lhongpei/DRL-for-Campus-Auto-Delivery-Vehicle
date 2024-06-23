from models.dqn import DQN, Double_DQN, D3QN, Dueling_DQN, ReplayBuffer
from env.wei_obs_grid import WeightedObsGrid
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import gym
from tqdm import tqdm
import wandb
import os
import pickle
from utils.utils import uniform_weight, normal_weight
use_wandb = True
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if use_wandb:
    wandb.init(project="wei_shortest")

grid_size = (10, 10)
lr = 1e-3
num_episodes = 5000
case_num = 1
hidden_dim = 3
gamma = 0.9
epsilon = 0.2
epsilon_decay = 1
target_update = 50
buffer_size = 10000
minimal_size = 500
batch_size = 256
reset_interval = 200  # 新增的超参数，设置每隔多少个 episode 重置一次环境
max_steps = 500  # 设置每个 episode 的最大步数，防止死循环
seed =42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

replay_buffer = ReplayBuffer(buffer_size)
state_dim = (4, grid_size[0], grid_size[1])
action_dim = 4  # Assuming there are 4 possible actions (up, down, left, right)

agent = Double_DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, epsilon_decay, target_update, device)
return_list = []
agent.load_state_dict(torch.load('saved_model/Double_DQN_random_start.pth'))
def create_new_env():
    return WeightedObsGrid(
        grid_size=grid_size,
        start=(0, 0),
        goal=(3, 2),
    )

env = create_new_env()


with tqdm(total=int(num_episodes), desc='Iteration ' ) as pbar:
    for i_episode in range(int(num_episodes)):
        episode_return = 0
        state = env.reset()
        done = False
        print('epsilon:', agent.epsilon)
        iter_num = 0

        while (not done) and (iter_num < max_steps):  # 添加最大步数限制
            iter_num += 1
            eligibles = env.eligible_action_idxes()
            if len(eligibles) == 0:
                break
            action = agent.take_action(state, eligibles)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            #print('state:', env.state['current_position'])
            episode_return += reward

            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)

        print(episode_return)
        if use_wandb:
            wandb.log({'return': episode_return, 'steps': env.total_steps, 'cost': env.total_cost})
        return_list.append(episode_return)

        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({
                'episode': '%d' % (i_episode + 1),
                'return': '%.3f' % np.mean(return_list[-10:])
            })
        
        # 定期重置环境
        if (i_episode + 1) % reset_interval == 0:
            env.reset(start_random=True)
            torch.save(agent.state_dict(), f'saved_model/model_{i_episode + 1}.pth')
        
        pbar.update(1)
        agent.epsilon_update()

    plt.plot(return_list)
    plt.savefig('train.png')


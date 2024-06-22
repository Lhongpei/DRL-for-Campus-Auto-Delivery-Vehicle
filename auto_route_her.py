from models.dqn import DQN, Double_DQN, D3QN, Dueling_DQN, ReplayBuffer
from models.her import Trajectory, ReplayBuffer_Trajectory
from env.wei_obs_grid import WeightedObsGrid
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import gym
from tqdm import tqdm
import wandb
import os
from utils.utils import uniform_weight, normal_weight
from utils.utils import NormalWeightGrid
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
epsilon_explore = 0.99
epsilon = 0.3
epsilon_decay = 1
epochs = 20
target_update = 50
buffer_size = 10000
minimal_size = 50
batch_size = 256
reset_interval = 200  # 新增的超参数，设置每隔多少个 episode 重置一次环境
max_steps = 500  # 设置每个 episode 的最大步数，防止死循环
seed =42
import pickle
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def create_new_env():
    return WeightedObsGrid(
        grid_size=grid_size,
        start=(0, 9),
        goal=(3, 2),
        obstacles=[(4,4), (1,1), (2,2), (3,3)],
        weights=np.random.rand(grid_size[0], grid_size[1]),
        dis_reward=False,
        wei_reset=NormalWeightGrid(grid_size),
        #goal_set=[(3, 2), (3, 6), (6, 2),(9, 9)]
    )

env = create_new_env()
env = pickle.load(open('env.pkl', 'rb'))
env.dis_reward = False

replay_buffer = ReplayBuffer_Trajectory(buffer_size, goal_reward = 2 * env.grid_size[0])
state_dim = (4, env.grid_size[0], env.grid_size[1])
action_dim = 4  # Assuming there are 4 possible actions (up, down, left, right)

agent = D3QN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon_explore, epsilon_decay, target_update, device)
return_list = []




with tqdm(total=int(num_episodes), desc='Iteration ' ) as pbar:
    for i_episode in range(int(num_episodes)):
        episode_return = 0
        state = env.reset(end_random=True, start_random=True)
        done = False
        print('epsilon:', agent.epsilon)
        iter_num = 0
        traj = Trajectory(state)
        while (not done) and (iter_num < max_steps):  # 添加最大步数限制
            iter_num += 1
            eligibles = env.eligible_action_idxes()
            if len(eligibles) == 0:
                break
            action = agent.take_action(state, eligibles)
            next_state, reward, done, _ = env.step(action)
            traj.store_step(action, next_state, reward, done)
            state = next_state
            #print('state:', env.state['current_position'])
            episode_return += reward
        replay_buffer.add_trajectory(traj)
        return_list.append(episode_return)
        
        print('episode return', episode_return)
        if use_wandb:
            wandb.log({'return': episode_return, 'steps': env.total_steps, 'cost': env.total_cost})
        if replay_buffer.size() > minimal_size:
            agent.epsilon = epsilon
            for _ in range(epochs):
                # if done:
                #     batch = replay_buffer.sample(batch_size, use_her=False)
                # else:
                batch = replay_buffer.sample(batch_size, use_her=True)
                agent.update(batch)

        if (i_episode + 1) % 10 == 0:
            pbar.set_postfix({
                'episode': '%d' % (i_episode + 1),
                'return': '%.3f' % np.mean(return_list[-10:])
            })
        
        # 定期重置环境 保存模型
        if (i_episode + 1) % reset_interval == 0:
            #env.reset(start_random=True)
            torch.save(agent.state_dict(), f'saved_model/model_{i_episode + 1}.pth')
        
        pbar.update(1)
        agent.epsilon_update()

    plt.plot(return_list)
    plt.savefig('train.png')


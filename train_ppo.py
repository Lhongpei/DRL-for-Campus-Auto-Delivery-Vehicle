
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from tqdm import tqdm
import wandb
import os
import gym
from models.ppo import PPO
def train_ppo(env, actor_lr, critic_lr, num_episodes, hidden_dim,
              gamma, epsilon, lmbda, epochs, reset_interval, max_steps, use_wandb=True):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if use_wandb:
        wandb.init(project="wei_shortest")

    state_dim = (4, env.grid_size[0], env.grid_size[1])
    action_dim = 4  # Assuming there are 4 possible actions (up, down, left, right)

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, epsilon, lmbda, epochs, device)
    return_list = []

    with tqdm(total=int(num_episodes), desc='Iteration ' ) as pbar:
        for i_episode in range(int(num_episodes)):
            episode_return = 0
            state = env.reset()
            done = False
            print('epsilon:', agent.epsilon)
            iter_num = 0
            transition_dict = {
                        'states': [],
                        'actions': [],
                        'next_states': [],
                        'rewards': [],
                        'dones': [],
                        'eligible': [],
                    }
            while (not done) and (iter_num < max_steps):  # 添加最大步数限制
                iter_num += 1
                eligibles = env.eligible_action_idxes()
                if len(eligibles) == 0:
                    break
                action = agent.take_action(state, eligibles)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['eligible'].append(eligibles)
                state = next_state
                episode_return += reward

            transition_dict['states'] = np.stack(transition_dict['states'])
            transition_dict['actions'] = np.stack(transition_dict['actions'])
            transition_dict['next_states'] = np.stack(transition_dict['next_states'])
            transition_dict['rewards'] = np.array(transition_dict['rewards'])
            transition_dict['dones'] = np.array(transition_dict['dones'])
            transition_dict['eligible'] = np.stack(transition_dict['eligible'])
            agent.update(transition_dict)
            return_list.append(episode_return)
            print('episode_return', episode_return)
            if use_wandb:
                wandb.log({'return': episode_return, 'steps': env.total_steps, 'cost': env.total_cost})

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })

            # 定期重置环境
            if (i_episode + 1) % max(10, reset_interval) == 0:
                env.reset(start_random=True, wei_reset=None, end_random=True)
                print('New Env:', env.start, env.goal)
                torch.save(agent.state_dict(), f'saved_model/model_{i_episode + 1}.pth')

            pbar.update(1)

        plt.plot(return_list)
        plt.savefig('train.png')
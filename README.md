# Introduction

This is a project trying to use DRL and MILP to implement a model for Auto-Delivery Vehicle.

## Model Implementation

All models are implemented under the folder named "models".

- The experiment with no crowding rates and fixed start and end points is implemented in the file "Qlearning_Sarsa.ipynb".
- The training part of DQN is implemented in the file "auto_route_dqn.py".
- The training part of PPO is implemented in the file "auto_route_ppo.py".
- The training part of D3QN-based HER is implemented in the file "auto_route_her.py".

## Comparison with Dijkstra

- The small case comparison with Dijkstra is implemented in the file "compared_dijkstra_small.ipynb".
- The large case comparison with Dijkstra is implemented in the file "compared_dijkstra_large.ipynb".

## Workflow Demonstration

The whole workflow demonstration is implemented in the file "milp_drl_demo.ipynb".

## MILP Implementation

My MILP implementation can be found in the folder named "milp_code". It is based on the Cardinal Optimizer (COPT), which doesn't require a license due to the small scale of my experiment.

## Download Link

[Download Cardinal Optimizer (COPT)]([link](https://www.shanshu.ai/copt))


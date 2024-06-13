from models.dqn import DQN
from env.wei_obs_grid import WeightedObsGrid
import matplotlib.pyplot as plt
import numpy

grid_size = (10, 10)
env = WeightedObsGrid(
        grid_size=grid_size,
        start=(0, 0),
        goal=(9, 9),
        obstacles=[(1, 1), (2, 2), (3, 3), (4, 4)],
        weights=numpy.random.rand(grid_size[0], grid_size[1])
    )
env.render()
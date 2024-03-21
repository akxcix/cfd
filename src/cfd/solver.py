import numpy as np

class Solver:
    def __init__(self, dt, total_steps, grid) -> None:
        self.total_steps = total_steps
        self.dt = dt
        self.grid = grid

    def step_once(self, grid):
        new_grid = np.zeros(grid.shape)
        # do stuff




import numpy as np
from .grids import Grid, GridPoint

class Solver:
    def __init__(self, dt, total_steps, grid) -> None:
        self.total_steps = total_steps
        self.dt = dt
        self.grid = grid
        self.new_grid = None
        self.k = 1.4

    def gauss_seidel(self):
        pass

    def step_once(self):
        m, n = self.grid.m, self.grid.n
        self.new_grid = Grid(m, n)
    
        for _ in range(10):
            for y in range(1,self.grid.m-1):
                for x in range(1,self.grid.n-1):
                    self.update_rho(x, y)
                    
                
        self.grid = self.new_grid

    def update_rho(self, x, y):
        index = y * self.grid.n + x

        rho = self.new_grid.get_at(x+1, y).rho
        rho += self.new_grid.get_at(x-1, y).rho
        rho += self.new_grid.get_at(x, y+1).rho
        rho += self.new_grid.get_at(x, y-1).rho
        
        rho = rho * self.k /4
        rho += self.grid.get_at(x,y).rho
        rho = rho/(1+self.k)

        self.new_grid.grid[index].rho = rho

    def run_simulation(self):
        for i in range(self.total_steps):
            self.step_once()
            if i % 50 == 0:
                self.grid.show_grid()




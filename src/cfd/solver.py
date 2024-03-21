import numpy as np
from .grids import Grid

class Solver:
    def __init__(self, dt, total_steps, grid) -> None:
        self.total_steps = total_steps
        self.dt = dt
        self.grid = grid
        self.new_grid = None
        self.k = 1.4
        self.gauss_seidel_iters = 15

    def step_once(self, dt):
        m, n = self.grid.m, self.grid.n
        self.new_grid = Grid(m, n)
    
        for _ in range(self.gauss_seidel_iters):
            for y in range(1,self.grid.m-1):
                for x in range(1,self.grid.n-1):
                    self.update_rho(x, y, dt)
                    
                
        self.grid = self.new_grid

    def update_rho(self, x, y, dt):
        index = y * self.grid.n + x

        newpoint = self.new_grid.get_at(x+1, y)
        newpoint += self.new_grid.get_at(x-1, y)
        newpoint += self.new_grid.get_at(x, y+1)
        newpoint += self.new_grid.get_at(x, y-1)
        
        newpoint = newpoint * (dt / 4)
        newpoint.rho = newpoint.rho * self.k
        newpoint += self.grid.get_at(x,y)
        newpoint.rho = newpoint.rho/(1+self.k)

        self.new_grid.grid[index] = newpoint

        # rho = self.new_grid.get_at(x+1, y).rho
        # rho += self.new_grid.get_at(x-1, y).rho
        # rho += self.new_grid.get_at(x, y+1).rho
        # rho += self.new_grid.get_at(x, y-1).rho
        
        # rho = rho * self.k /4
        # rho += self.grid.get_at(x,y).rho
        # rho = rho/(1+self.k)

        # self.new_grid.grid[index].rho = rho

    def run_simulation(self):
        for i in range(self.total_steps):
            self.step_once(self.dt)
            if i % 10 == 0:
                print(sum([gp.rho for gp in self.grid.grid]))
                self.grid.show_grid()




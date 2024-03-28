import numpy as np
from .grids import Grid, GridPoint

class Solver:
    def __init__(self, dt, total_steps, grid, coeff_diff = 0.1, coeff_visc = 0.1, gauss_seidel_iters=10) -> None:
        self.grid = grid
        self.new_grid = None

        self.dt = dt

        self.coeff_diff = coeff_diff
        self.coeff_visc = coeff_visc

        self.total_steps = total_steps
        self.gauss_seidel_iters = gauss_seidel_iters

    def step_once(self, dt):
        m, n = self.grid.m, self.grid.n
        self.new_grid = Grid(m, n)
    
        for _ in range(self.gauss_seidel_iters):
            for y in range(1,self.grid.m-1):
                for x in range(1,self.grid.n-1):
                    self.diffuse(x, y, dt)

        self.set_bnd()
                    
                
        self.grid = self.new_grid

    def set_bnd(self):
        m, n = self.grid.m, self.grid.n

        # Left and right boundaries
        for y in range(1, m - 1):
            self.new_grid.set_at(0, y, self.grid.get_at(1, y))
            self.new_grid.set_at(n - 1, y, self.grid.get_at(n - 2, y))

        # Bottom and top boundaries
        for x in range(1, n - 1):
            self.new_grid.set_at(x, 0, self.grid.get_at(x, 1))
            self.new_grid.set_at(x, m - 1, self.grid.get_at(x, m - 2))

        # Corner cases
        self.new_grid.set_at(0, 0, self.grid.get_at(1, 1))
        self.new_grid.set_at(n - 1, 0, self.grid.get_at(n - 2, 1))
        self.new_grid.set_at(0, m - 1, self.grid.get_at(1, m - 2))
        self.new_grid.set_at(n - 1, m - 1, self.grid.get_at(n - 2, m - 2))

    def add_density(self,x,y,amount):
        point = self.grid.get_at(x,y)
        point.rho += amount
        self.grid.set_at(x,y,point)

    def diffuse(self, x, y, dt):
        a = dt*self.coeff_diff*self.grid.m*self.grid.n
        index = y * self.grid.n + x

        newpoint = GridPoint()
        newpoint += self.grid.get_at(x, y)

        newpoint += (
            self.grid.get_at(x+1, y) 
            + self.grid.get_at(x-1, y)
            + self.grid.get_at(x, y+1)
            + self.grid.get_at(x, y-1)
        ) * a

        newpoint = newpoint * (1/(1+4*a))

        self.new_grid.grid[index] = newpoint

    def advect_density(self):
        pass

    def advect_velocity(self):
        pass

    def run_simulation(self):
        for i in range(self.total_steps):
            self.step_once(self.dt)
            if i % 10 == 0:
                print(sum([gp.rho for gp in self.grid.grid]))
                self.grid.show_grid()




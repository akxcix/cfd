import numpy as np

from .grids import Grid, GridPoint


class Solver:
    """
    implements the paper "Real Time Fluid Dynamic" by jos stam.
    reference implementation from Mike Ash: https://mikeash.com/pyblog/fluid-simulation-for-dummies.html
    """
    def __init__(
        self,
        dt: float,
        total_steps: int,
        grid: Grid,
        coeff_diff: float = 0.1,
        coeff_visc: float = 0.1,
        gauss_seidel_iters: int = 10,
        plot_interval: int = 10,
    ) -> None:
        self.grid = grid
        self._new_grid: Grid | None = None

        self.dt = dt

        self.coeff_diff = coeff_diff
        self.coeff_visc = coeff_visc

        self.total_steps = total_steps
        self.gauss_seidel_iters = gauss_seidel_iters
        self.plot_interval = plot_interval

        # minmax
        self.min_rho =0
        self.max_rho = 0

        # specific to mike ash's immplementation
        self.grid2arr()

    def grid2arr(self):
        self.density = [p.rho for p in self.grid.grid]
        self.s = self.density.copy()
        
        self.ux0 = [p.ux for p in self.grid.grid]
        self.uy0 = [p.uy for p in self.grid.grid]

        self.ux = self.ux0.copy()
        self.uy = self.uy0.copy()

        self.min_rho = min(self.density)
        self.max_rho = max(self.density)

    def arr2grid(self):
        for i in range(len(self.grid.grid)):
            self.grid.grid[i].rho = self.density[i]
            self.grid.grid[i].ux = self.ux0[i]
            self.grid.grid[i].uy = self.uy0[i]

    def get_index(self, x, y):
        idx = int(y * self.grid.m + x)

        return idx

    def step_once(self, dt):
        self.diffuse(1, self.ux0, self.ux, self.coeff_visc, dt)
        self.diffuse(2, self.uy0, self.uy, self.coeff_visc, dt)
        
        self.project(self.ux0, self.uy0, self.ux, self.uy)
        
        self.advect(1, self.ux, self.ux0, self.ux0, self.uy0, dt)
        self.advect(2, self.uy, self.uy0, self.ux0, self.uy0, dt)
        
        self.project(self.ux, self.uy, self.ux0, self.uy0);
        
        self.diffuse(0, self.s, self.density, self.coeff_diff, dt);
        self.advect(0, self.density, self.s, self.ux, self.uy, dt);

        self.arr2grid()
    


    def set_bnd(self, b, x):
        m = self.grid.m
        n = self.grid.n

        for i in range(1,m-1):
            x[self.get_index(i, 0)] = -x[self.get_index(i, 1)] if b == 2 else x[self.get_index(i, 1)]
            x[self.get_index(i, n-1)] = -x[self.get_index(i, n-2)] if b == 2 else x[self.get_index(i, n-2)]  

        for j in range(1,n-1):
            x[self.get_index(0, j)] = -x[self.get_index(1, j)] if b == 1 else x[self.get_index(1, j)]
            x[self.get_index(m-1, j)] = -x[self.get_index(m-2, j)] if b == 1 else x[self.get_index(m-2, j)]

        x[self.get_index(0, 0)] = 0.5 * (x[self.get_index(1, 0)] + x[self.get_index(0, 1)])
        x[self.get_index(0, n-1)] = 0.5 * (x[self.get_index(1, n-1)] + x[self.get_index(0, n-2)])
        x[self.get_index(m-1, 0)] = 0.5 * (x[self.get_index(m-2, 0)] + x[self.get_index(m-1, 1)])
        x[self.get_index(m-1, n-1)] = 0.5 * (x[self.get_index(m-2, n-1)] + x[self.get_index(m-1, n-2)])

        return x
    
    def lin_solve(self, b, x, x0, a, c):
        m = self.grid.m
        n = self.grid.n
        c_inv = 1.0 / c

        for iter in range(self.gauss_seidel_iters):
            for j in range(1,n-1):
                for i in range(1,m-1):
                    x[self.get_index(i, j)] = (x0[self.get_index(i, j)]
                            + a*(
                                x[self.get_index(i+1, j)]
                                +x[self.get_index(i-1, j)]
                                +x[self.get_index(i, j+1)]
                                +x[self.get_index(i, j-1)]
                        )) * c_inv

            self.set_bnd(b, x)

    def diffuse(self, b, x, x0, diff, dt):
        m = self.grid.m
        n = self.grid.n

        a = dt * diff * (m - 2) * (n - 2);
        self.lin_solve(b, x, x0, a, 1 + 4 * a);
    
    def project(self, velocX, velocY, p, div):
        m = self.grid.m
        n = self.grid.n

        for j in range(1,n-1):
            for i in range(1,m-1):
                div[self.get_index(i, j)] = -0.5 *(
                        velocX[self.get_index(i+1, j)]
                        -velocX[self.get_index(i-1, j)]
                        +velocY[self.get_index(i  , j+1)]
                        -velocY[self.get_index(i  , j-1)]
                    )/n;
                p[self.get_index(i, j)] = 0
            
        
        self.set_bnd(0, div) 
        self.set_bnd(0, p)
        self.lin_solve(0, p, div, 1, 4)
        
        for j in range(1,n-1):
            for i in range(1,m-1):
                velocX[self.get_index(i, j)] -= 0.5 * (p[self.get_index(i+1, j)]
                                                -p[self.get_index(i-1, j)]) * self.grid.m
                velocY[self.get_index(i, j)] -= 0.5 * (p[self.get_index(i, j+1)]
                                                -p[self.get_index(i, j-1)]) * self.grid.n
               
        self.set_bnd(1, velocX)
        self.set_bnd(2, velocY)

    def advect(self, b, d, d0, velocX, velocY, dt):
        m = self.grid.m
        n = self.grid.n
        
        dtx = dt * (m - 2)
        dty = dt * (n - 2)
        
                
        for j in range(1,n-1):
            for i in range(1,m-1):
                tmp1 = dtx * velocX[self.get_index(i, j)]
                tmp2 = dty * velocY[self.get_index(i, j)]
                x    = i - tmp1 
                y    = j - tmp2
                
                if x < 0.5:
                    x = 0.5
                if x > m + 0.5:
                    x = m + 0.5
                i0 = int(x)
                i1 = i0 + 1

                if y < 0.5:
                    y = 0.5
                if y > n + 0.5:
                    y = n + 0.5
                j0 = int(y)
                j1 = j0 + 1.0
                
                
                s1 = x - i0
                s0 = 1.0 - s1 
                t1 = y - j0
                t0 = 1.0 - t1
                
                d[self.get_index(i, j)] = s0*(t0*d0[self.get_index(i0,j0)]+t1*d0[self.get_index(i0,j1)])+s1*(t0*d0[self.get_index(i1,j0)]+t1*d0[self.get_index(i1,j1)])
                
        self.set_bnd(b, d)

    # def _set_bnd(self, attr):
    #     m, n = self.grid.m, self.grid.n

    #     assert self.new_grid

    #     # Left and right boundaries
    #     for y in range(1, m - 1):
    #         self.new_grid[0, y] = self.grid[1, y]
    #         self.new_grid[n - 1, y] = self.grid[n - 2, y]

    #     # Bottom and top boundaries
    #     for x in range(1, n - 1):
    #         self.new_grid[x, 0] = self.grid[x, 1]
    #         self.new_grid[x, m - 1] = self.grid[x, m - 2]

    #     # Corner cases
    #     self.new_grid[0, 0] = self.grid[1, 1]
    #     self.new_grid[n - 1, 0] = self.grid[n - 2, 1]
    #     self.new_grid[0, m - 1] = self.grid[1, m - 2]
    #     self.new_grid[n - 1, m - 1] = self.grid[n - 2, m - 2]

    # def add_density(self, x: int, y: int, amount: float):
    #     point = self.grid[x, y]
    #     point.rho += amount
    #     self.grid[x, y] = point

    # def lin_solve(self, attr, a, c):
    #     c_inv = 1/c
    #     for k in range(self.gauss_seidel_iters):
    #         for y in range(self.m):
    #             for x in range(self.n):
    #                 newpoint = GridPoint()
    #                 newpoint.__setattr__(attr, (self.grid._get_at(x,y).__getattribute__(attr) + a * (
    #                     self.grid._get_at(x+1, y).__getattribute__(attr)
    #                     +self.grid._get_at(x-1, y).__getattribute__(attr)
    #                     +self.grid._get_at(x, y+1).__getattribute__(attr)
    #                     +self.grid._get_at(x, y-1).__getattribute__(attr)
    #                 )) * c_inv)
    #                 self.new_grid.__setattr__(attr, newpoint.__getattribute__(attr))

    #         self._set_bnd(attr)

    # def diffuse_velocity(self):
    #     a = self.dt * self.coeff_diff * (self.m-2) * (self.n -2)
    #     self.lin_solve("ux", a, 1 + 4*a)
    #     self.lin_solve("uy", a, 1 + 4*a)

    # def project(self):
    #     div = [0 for _ in range(self.m * self.n)]
    #     p = [0 for _ in range(self.m * self.n)]

    #     for x in range(self.n):
    #         for y in range(self.m):
    #             div[self.grid._index(x, y)] = -0.5 * (self.grid)

    def run_simulation(self):
        grids = []
        for i in range(self.total_steps):
            self.step_once(self.dt)
            if i % self.plot_interval == 0:
                print(sum([gp.rho for gp in self.grid.grid]))
                self.grid.show_grid(self.min_rho, self.max_rho)
                # grids.append(self.grid.grid)

        return grids

    def _new_grid_getter(self) -> Grid:
        assert self._new_grid
        return self._new_grid

    def _new_grid_setter(self, value: Grid):
        self._new_grid = value

    new_grid = property(_new_grid_getter, _new_grid_setter)

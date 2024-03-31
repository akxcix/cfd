import numpy as np

from .grids import Grid, GridPoint


class Solver:
    def __init__(
        self,
        dt: float,
        total_steps: int,
        grid: Grid,
        coeff_diff: float = 0.1,
        coeff_visc: float = 0.1,
        gauss_seidel_iters: int = 10,
    ) -> None:
        self.grid = grid
        self._new_grid: Grid | None = None

        self.dt = dt

        self.coeff_diff = coeff_diff
        self.coeff_visc = coeff_visc

        self.total_steps = total_steps
        self.gauss_seidel_iters = gauss_seidel_iters

    def step_once(self, dt):
        m, n = self.grid.m, self.grid.n
        self.new_grid = Grid(m, n)

        for _ in range(self.gauss_seidel_iters):
            for y in range(1, self.grid.m - 1):
                for x in range(1, self.grid.n - 1):
                    self.diffuse(x, y, dt)

        self._set_bnd()

        self.grid = self.new_grid

    def _set_bnd(self):
        m, n = self.grid.m, self.grid.n

        assert self.new_grid

        # Left and right boundaries
        for y in range(1, m - 1):
            self.new_grid[0, y] = self.grid[1, y]
            self.new_grid[n - 1, y] = self.grid[n - 2, y]

        # Bottom and top boundaries
        for x in range(1, n - 1):
            self.new_grid[x, 0] = self.grid[x, 1]
            self.new_grid[x, m - 1] = self.grid[x, m - 2]

        # Corner cases
        self.new_grid[0, 0] = self.grid[1, 1]
        self.new_grid[n - 1, 0] = self.grid[n - 2, 1]
        self.new_grid[0, m - 1] = self.grid[1, m - 2]
        self.new_grid[n - 1, m - 1] = self.grid[n - 2, m - 2]

    def add_density(self, x: int, y: int, amount: float):
        point = self.grid[x, y]
        point.rho += amount
        self.grid[x, y] = point

    def diffuse(self, x: int, y: int, dt: float):
        a = dt * self.coeff_diff * self.grid.m * self.grid.n
        index = y * self.grid.n + x

        newpoint = GridPoint()
        newpoint += self.grid[x, y]

        newpoint += (
            self.grid[x + 1, y]
            + self.grid[x - 1, y]
            + self.grid[x, y + 1]
            + self.grid[x, y - 1]
        ) * (a / (1 + 4 * a))

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

    def _new_grid_getter(self) -> Grid:
        assert self._new_grid
        return self._new_grid

    def _new_grid_setter(self, value: Grid):
        self._new_grid = value

    new_grid = property(_new_grid_getter, _new_grid_setter)

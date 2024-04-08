import time

import numba as nb
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
        self.min_rho = 0
        self.max_rho = 0

        # specific to mike ash's immplementation
        self.grid2arr()

    def grid2arr(self):
        self.density = np.array([p.rho for p in self.grid.grid]).astype("f")
        self.s = self.density.copy()

        self.ux0 = np.array([p.ux for p in self.grid.grid]).astype("f")
        self.uy0 = np.array([p.uy for p in self.grid.grid]).astype("f")

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
        return index_jit(self.grid.m, self.grid.n, x, y)

    def step_once(self, dt):
        self.diffuse(1, self.ux0, self.ux, self.coeff_visc, dt)
        self.diffuse(2, self.uy0, self.uy, self.coeff_visc, dt)

        self.project(self.ux0, self.uy0, self.ux, self.uy)

        self.advect(1, self.ux, self.ux0, self.ux0, self.uy0, dt)
        self.advect(2, self.uy, self.uy0, self.ux0, self.uy0, dt)

        self.project(self.ux, self.uy, self.ux0, self.uy0)

        self.diffuse(0, self.s, self.density, self.coeff_diff, dt)
        self.advect(0, self.density, self.s, self.ux, self.uy, dt)

        self.arr2grid()

    def lin_solve(self, b, x, x0, a, c):
        lin_solve_jit(self.grid.m, self.grid.n, b, x, x0, a, c, self.gauss_seidel_iters)

    def diffuse(self, b, x, x0, diff, dt):
        m = self.grid.m
        n = self.grid.n

        a = dt * diff * (m - 2) * (n - 2)
        self.lin_solve(b, x, x0, a, 1 + 4 * a)

    def project(self, velocX, velocY, p, div):
        project_jit(
            self.grid.m, self.grid.n, velocX, velocY, p, div, self.gauss_seidel_iters
        )

    def advect(self, b, d, d0, velocX, velocY, dt):
        advect_jit(self.grid.m, self.grid.n, b, d, d0, velocX, velocY, dt)

    def run_simulation(self):
        grids = []
        for i in range(self.total_steps):
            start = time.time()

            self.step_once(self.dt)

            end = time.time()

            print(f"step {i} took {end-start} seconds")

            if (i + 1) % self.plot_interval == 0:
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

    @property
    def m(self) -> int:
        return self.grid.m

    @property
    def n(self) -> int:
        return self.grid.n


@nb.jit
def advect_jit(m, n, b, d, d0, velocX, velocY, dt):
    dtx = dt * (m - 2)
    dty = dt * (n - 2)

    for j in range(1, n - 1):
        for i in range(1, m - 1):
            tmp1 = dtx * velocX[index_jit(m, n, i, j)]
            tmp2 = dty * velocY[index_jit(m, n, i, j)]
            x = i - tmp1
            y = j - tmp2

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

            d[index_jit(m, n, i, j)] = s0 * (
                t0 * d0[index_jit(m, n, i0, j0)] + t1 * d0[index_jit(m, n, i0, j1)]
            ) + s1 * (
                t0 * d0[index_jit(m, n, i1, j0)] + t1 * d0[index_jit(m, n, i1, j1)]
            )

    set_bnd_jit(m, n, b, d)


@nb.jit
def index_jit(m, n, x, y):
    if x >= m:
        x = m - 1

    if y >= n:
        y = n - 1

    return int(y * m + x)


@nb.jit
def set_bnd_jit(m, n, b, x):
    for i in range(1, m - 1):
        x[index_jit(m, n, i, 0)] = (
            -x[index_jit(m, n, i, 1)] if b == 2 else x[index_jit(m, n, i, 1)]
        )
        x[index_jit(m, n, i, n - 1)] = (
            -x[index_jit(m, n, i, n - 2)] if b == 2 else x[index_jit(m, n, i, n - 2)]
        )

    for j in range(1, n - 1):
        x[index_jit(m, n, 0, j)] = (
            -x[index_jit(m, n, 1, j)] if b == 1 else x[index_jit(m, n, 1, j)]
        )
        x[index_jit(m, n, m - 1, j)] = (
            -x[index_jit(m, n, m - 2, j)] if b == 1 else x[index_jit(m, n, m - 2, j)]
        )

    x[index_jit(m, n, 0, 0)] = 0.5 * (
        x[index_jit(m, n, 1, 0)] + x[index_jit(m, n, 0, 1)]
    )
    x[index_jit(m, n, 0, n - 1)] = 0.5 * (
        x[index_jit(m, n, 1, n - 1)] + x[index_jit(m, n, 0, n - 2)]
    )
    x[index_jit(m, n, m - 1, 0)] = 0.5 * (
        x[index_jit(m, n, m - 2, 0)] + x[index_jit(m, n, m - 1, 1)]
    )
    x[index_jit(m, n, m - 1, n - 1)] = 0.5 * (
        x[index_jit(m, n, m - 2, n - 1)] + x[index_jit(m, n, m - 1, n - 2)]
    )


@nb.jit
def lin_solve_jit(m, n, b, x, x0, a, c, gauss_seidel_iters):
    c_inv = 1.0 / c

    for _ in range(gauss_seidel_iters):
        for j in range(1, n - 1):
            for i in range(1, m - 1):
                x[index_jit(m, n, i, j)] = (
                    x0[index_jit(m, n, i, j)]
                    + a
                    * (
                        x[index_jit(m, n, i + 1, j)]
                        + x[index_jit(m, n, i - 1, j)]
                        + x[index_jit(m, n, i, j + 1)]
                        + x[index_jit(m, n, i, j - 1)]
                    )
                ) * c_inv

        set_bnd_jit(m, n, b, x)


@nb.jit
def project_jit(m, n, velocX, velocY, p, div, gauss_seidel_iters):
    for j in range(1, n - 1):
        for i in range(1, m - 1):
            div[index_jit(m, n, i, j)] = (
                -0.5
                * (
                    velocX[index_jit(m, n, i + 1, j)]
                    - velocX[index_jit(m, n, i - 1, j)]
                    + velocY[index_jit(m, n, i, j + 1)]
                    - velocY[index_jit(m, n, i, j - 1)]
                )
                / n
            )
            p[index_jit(m, n, i, j)] = 0

    set_bnd_jit(m, n, 0, div)
    set_bnd_jit(m, n, 0, p)
    lin_solve_jit(m, n, 0, p, div, 1, 4, gauss_seidel_iters)

    for j in range(1, n - 1):
        for i in range(1, m - 1):
            velocX[index_jit(m, n, i, j)] -= (
                0.5 * (p[index_jit(m, n, i + 1, j)] - p[index_jit(m, n, i - 1, j)]) * m
            )
            velocY[index_jit(m, n, i, j)] -= (
                0.5 * (p[index_jit(m, n, i, j + 1)] - p[index_jit(m, n, i, j - 1)]) * n
            )

    set_bnd_jit(m, n, 1, velocX)
    set_bnd_jit(m, n, 2, velocY)

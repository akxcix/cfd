import time

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

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
        self.density = np.array([p.rho for p in self.grid.grid], dtype="f").reshape(
            self.m, self.n
        )
        self.s = self.density.copy()

        self.ux0 = np.array([p.ux for p in self.grid.grid], dtype="f").reshape(
            self.m, self.n
        )
        self.uy0 = np.array([p.uy for p in self.grid.grid], dtype="f").reshape(
            self.m, self.n
        )

        self.ux = self.ux0.copy()
        self.uy = self.uy0.copy()

        self.min_rho = np.min(self.density)
        self.max_rho = np.max(self.density)

    def arr2grid(self):
        density = self.density.ravel()
        ux0 = self.ux0.ravel()
        uy0 = self.uy0.ravel()

        for i in range(len(self.grid.grid)):
            self.grid.grid[i].rho = density[i]
            self.grid.grid[i].ux = ux0[i]
            self.grid.grid[i].uy = uy0[i]

    def step_once(self, dt: float):
        """
        Perform one step of the fluid simulation.

        Parameters:
            dt: The time step
        """

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
        """
        Solve a linear system of equations.

        Parameters:
            b: The boundary condition
            x: The grid to solve
            x0: The initial grid
            a: The coefficient
            c: The diagonal coefficient
        """

        lin_solve_jit(self.m, self.n, b, x, x0, a, c, self.gauss_seidel_iters)

    def diffuse(self, b, x, x0, diff, dt):
        """
        Diffuse a quantity.

        Parameters:
            b: The boundary condition
            x: The grid to diffuse
            x0: The initial grid
            diff: The diffusion coefficient
            dt: The time step
        """

        a = dt * diff * (self.m - 2) * (self.n - 2)
        self.lin_solve(b, x, x0, a, 1 + 4 * a)

    def project(self, velocX, velocY, p, div):
        """
        Project the velocity field.

        Parameters:
            velocX: The x component of the velocity field
            velocY: The y component of the velocity field
            p: The pressure field
            div: The divergence field
        """
        project_jit(self.m, self.n, velocX, velocY, p, div, self.gauss_seidel_iters)

    def advect(self, b, d, d0, velocX, velocY, dt):
        advect_jit(self.m, self.n, b, d, d0, velocX, velocY, dt)

    def run_simulation(self):
        """
        Run the simulation for the given number of steps.

        Returns:
            A list of grids at each plot interval
        """

        grids = []
        for i in range(self.total_steps):
            start = time.time()

            self.step_once(self.dt)

            end = time.time()

            print(f"step {i} took {end-start} seconds")

            if (i + 1) % self.plot_interval == 0:
                print(sum([gp.rho for gp in self.grid.grid]))

                grids.append(self.grid.grid)

        return grids

    @property
    def m(self) -> int:
        return self.grid.m

    @property
    def n(self) -> int:
        return self.grid.n


@nb.jit
def advect_jit(m: int, n: int, b, d, d0, velocX, velocY, dt):
    """
    Advect a quantity. JITed.

    Parameters:
        m: The number of rows in the grid
        n: The number of columns in the grid
        b: The boundary condition
        d: The quantity to advect
        d0: The initial quantity
        velocX: The x component of the velocity field
        velocY: The y component of the velocity field
        dt: The time step
    """

    dtx = dt * (m - 2)
    dty = dt * (n - 2)

    for j in range(1, n - 1):
        for i in range(1, m - 1):
            tmp1 = dtx * velocX[i, j]
            tmp2 = dty * velocY[i, j]
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
            j1 = j0 + 1

            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            # HACK: i0, j0, i1, j1 are clipped

            if i0 >= m:
                i0 = m - 1

            if j0 >= n:
                j0 = n - 1

            if i1 >= m:
                i1 = m - 1

            if j1 >= n:
                j1 = n - 1

            dd0 = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1])
            dd1 = s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
            d[i, j] = dd0 + dd1

    set_bnd_jit(m, n, b, d)


@nb.jit
def set_bnd_jit(m: int, n: int, b: int, x: NDArray):
    """
    Set boundary conditions. JITed.

    Parameters:
        m: The number of rows in the grid
        n: The number of columns in the grid
        b: The boundary condition
        x: The grid
    """

    for i in range(1, m - 1):
        x[i, 0] = -x[i, 1] if b == 2 else x[i, 1]
        x[i, n - 1] = -x[i, n - 2] if b == 2 else x[i, n - 2]

    for j in range(1, n - 1):
        x[0, j] = -x[1, j] if b == 1 else x[1, j]
        x[m - 1, j] = -x[m - 2, j] if b == 1 else x[m - 2, j]

    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, n - 1] = 0.5 * (x[1, n - 1] + x[0, n - 2])
    x[m - 1, 0] = 0.5 * (x[m - 2, 0] + x[m - 1, 1])
    x[m - 1, n - 1] = (x[m - 2, n - 1] + x[m - 1, n - 2]) / 2


@nb.jit
def lin_solve_jit(
    m: int,
    n: int,
    b: int,
    x: NDArray,
    x0: NDArray,
    a: float,
    c: float,
    gauss_seidel_iters: int,
):
    """
    Solve a linear system of equations. JITed.

    Parameters:
        m: The number of rows in the grid
        n: The number of columns in the grid
        b: The boundary condition
        x: The grid to solve
        x0: The initial grid
        a: The coefficient
        c: The diagonal coefficient
        gauss_seidel_iters: The number of Gauss-Seidel iterations
    """

    c_inv = 1.0 / c

    for _ in range(gauss_seidel_iters):
        for j in range(1, n - 1):
            for i in range(1, m - 1):
                x[i, j] = (
                    x0[i, j]
                    + a * (x[i + 1, j] + x[i - 1, j] + x[i, j + 1] + x[i, j - 1])
                ) * c_inv

        set_bnd_jit(m, n, b, x)


@nb.jit
def project_jit(m: int, n: int, velocX, velocY, p, div, gauss_seidel_iters):
    """
    Project the velocity field. JITed.

    Parameters:
        m: The number of rows in the grid
        n: The number of columns in the grid
        velocX: The x component of the velocity field
        velocY: The y component of the velocity field
        p: The pressure field
        div: The divergence field
        gauss_seidel_iters: The number of Gauss-Seidel iterations
    """
    for j in range(1, n - 1):
        for i in range(1, m - 1):
            div[i, j] = (
                -0.5
                * (
                    velocX[i + 1, j]
                    - velocX[i - 1, j]
                    + velocY[i, j + 1]
                    - velocY[i, j - 1]
                )
                / n
            )
            p[i, j] = 0

    set_bnd_jit(m, n, 0, div)
    set_bnd_jit(m, n, 0, p)
    lin_solve_jit(m, n, 0, p, div, 1, 4, gauss_seidel_iters)

    for j in range(1, n - 1):
        for i in range(1, m - 1):
            velocX[i, j] -= 0.5 * (p[i + 1, j] - p[i - 1, j]) * m
            velocY[i, j] -= 0.5 * (p[i, j + 1] - p[i, j - 1]) * n

    set_bnd_jit(m, n, 1, velocX)
    set_bnd_jit(m, n, 2, velocY)

import itertools

import alive_progress as ap
import numpy as np
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
        self.density = np.array([p.rho for p in self.grid.grid]).reshape(self.m, self.n)
        self.s = self.density.copy()

        self.ux0 = np.array([p.ux for p in self.grid.grid]).reshape(self.m, self.n)
        self.uy0 = np.array([p.uy for p in self.grid.grid]).reshape(self.m, self.n)

        self.ux = self.ux0.copy()
        self.uy = self.uy0.copy()

        self.min_rho = self.density.min()
        self.max_rho = self.density.max()

    def arr2grid(self):
        density = self.density.flatten()
        ux0 = self.ux0.flatten()
        uy0 = self.uy0.flatten()
        for i in range(len(self.grid.grid)):
            self.grid.grid[i].rho = density[i]
            self.grid.grid[i].ux = ux0[i]
            self.grid.grid[i].uy = uy0[i]

    def get_index(self, x, y):
        idx = int(y * self.grid.m + x)

        return idx

    def step_once(self, dt: float):
        self.diffuse(1, self.ux0, self.ux, self.coeff_visc, dt)
        self.diffuse(2, self.uy0, self.uy, self.coeff_visc, dt)

        self.project(self.ux0, self.uy0, self.ux, self.uy)

        self.advect(1, self.ux, self.ux0, self.ux0, self.uy0, dt)
        self.advect(2, self.uy, self.uy0, self.ux0, self.uy0, dt)

        self.project(self.ux, self.uy, self.ux0, self.uy0)

        self.diffuse(0, self.s, self.density, self.coeff_diff, dt)
        self.advect(0, self.density, self.s, self.ux, self.uy, dt)

        self.arr2grid()

    def set_bnd(self, b: float, x: NDArray):
        m = self.m
        n = self.n

        x[:, 0] = -x[:, 1] if b == 2 else x[:, 1]
        x[:, n - 1] = -x[:, n - 2] if b == 2 else x[:, n - 2]
        # for i in range(1, m - 1):
        #     x[i, 0] = -x[i, 1] if b == 2 else x[i, 1]
        #     x[i, n - 1] = -x[i, n - 2] if b == 2 else x[i, n - 2]

        x[0, :] = -x[1, :] if b == 1 else x[1, :]
        x[m - 1, :] = -x[m - 2, :] if b == 1 else x[m - 2, :]
        # for j in range(1, n - 1):
        #     x[0, j] = -x[1, j] if b == 1 else x[1, j]
        #     x[m - 1, j] = -x[m - 2, j] if b == 1 else x[m - 2, j]

        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, n - 1] = 0.5 * (x[1, n - 1] + x[0, n - 2])
        x[m - 1, 0] = 0.5 * (x[m - 2, 0] + x[m - 1, 1])
        x[m - 1, n - 1] = 0.5 * (x[m - 2, n - 1] + x[m - 1, n - 2])

        return x

    def lin_solve(self, b: int, x: NDArray, x0: NDArray, a: float, c: float):
        m = self.m
        n = self.n
        c_inv = 1.0 / c

        assert x.shape == (self.m, self.n), x.shape
        assert x0.shape == (self.m, self.n), x0.shape

        for _ in range(self.gauss_seidel_iters):
            x[1 : n - 1, 1 : m - 1] = (
                x0[1 : n - 1, 1 : m - 1]
                + a
                * (
                    x[2:, 1 : m - 1]
                    + x[0 : n - 2, 1 : m - 1]
                    + x[1 : n - 1, 2:]
                    + x[1 : n - 1, 0 : m - 2]
                )
            ) * c_inv

            # for j in range(1, n - 1):
            #     for i in range(1, m - 1):
            #         x[self.get_index(i, j)] = (
            #             x0[self.get_index(i, j)]
            #             + a
            #             * (
            #                 x[self.get_index(i + 1, j)]
            #                 + x[self.get_index(i - 1, j)]
            #                 + x[self.get_index(i, j + 1)]
            #                 + x[self.get_index(i, j - 1)]
            #             )
            #         ) * c_inv

            self.set_bnd(b, x)

    def diffuse(self, b: float, x: NDArray, x0: NDArray, diff: float, dt: float):
        m = self.m
        n = self.n

        a = dt * diff * (m - 2) * (n - 2)
        self.lin_solve(b, x, x0, a, 1 + 4 * a)

    def project(self, velocX: NDArray, velocY: NDArray, p: NDArray, div: NDArray):
        m = self.m
        n = self.n

        assert velocX.shape == (self.m, self.n)
        assert velocY.shape == (self.m, self.n)
        assert p.shape == (self.m, self.n)
        assert div.shape == (self.m, self.n)

        div[1 : m - 1, 1 : n - 1] = (
            -0.5
            * (
                velocX[2:, 1 : n - 1]
                - velocX[0 : m - 2, 1 : n - 1]
                + velocY[1 : m - 1, 2:]
                - velocY[1 : m - 1, 0 : n - 2]
            )
            / n
        )
        p[1 : m - 1, 1 : n - 1] = 0

        # for j in range(1, n - 1):
        #     for i in range(1, m - 1):
        #         div[i, j] = (
        #             -0.5
        #             * (
        #                 velocX[i + 1, j]
        #                 - velocX[i - 1, j]
        #                 + velocY[i, j + 1]
        #                 - velocY[i, j - 1]
        #             )
        #             / n
        #         )
        #         p[i, j] = 0

        self.set_bnd(0, div)
        self.set_bnd(0, p)
        self.lin_solve(0, p, div, 1, 4)

        velocX[1 : m - 1, 1 : n - 1] -= (
            0.5 * (p[2:m, 1 : n - 1] - p[0 : m - 2, 1 : n - 1]) * self.m
        )
        velocY[1 : m - 1, 1 : n - 1] -= (
            0.5 * (p[1 : m - 1, 2:n] - p[1 : m - 1, 0 : n - 2]) * self.n
        )
        # for j in range(1, n - 1):
        #     for i in range(1, m - 1):
        #         velocX[i, j] -= 0.5 * (p[i + 1, j] - p[i - 1, j]) * self.m
        #         velocY[i, j] -= 0.5 * (p[i, j + 1] - p[i, j - 1]) * self.n

        self.set_bnd(1, velocX)
        self.set_bnd(2, velocY)

    def advect(
        self,
        b: float,
        d: NDArray,
        d0: NDArray,
        velocX: NDArray,
        velocY: NDArray,
        dt: float,
    ):
        m = self.m
        n = self.n

        dtx = dt * (m - 2)
        dty = dt * (n - 2)

        tmp1 = dtx * velocX
        tmp2 = dty * velocY

        # Get all combinations of i, j
        idx = np.array(list(itertools.product(range(1, m - 1), range(1, n - 1))))

        x = idx[:, 0]
        y = idx[:, 1]
        x -= dtx * velocX[x, y]
        y -= dty * velocY[x, y]
        x = np.clip(x, 0.5, m + 0.5)
        y = np.clip(y, 0.5, n + 0.5)

        i0 = x.astype(int)
        i1 = i0 + 1
        j0 = y.astype(int)
        j1 = j0 + 1

        s1 = x - i0
        s0 = 1.0 - s1
        t1 = y - j0
        t0 = 1.0 - t1

        d[x, y] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (
            t0 * d0[i1, j0] + t1 * d0[i1, j1]
        )

        # for j in range(1, n - 1):
        #     for i in range(1, m - 1):
        #         tmp1 = dtx * velocX[i, j]
        #         tmp2 = dty * velocY[i, j]
        #         x = i - tmp1
        #         y = j - tmp2

        #         if x < 0.5:
        #             x = 0.5
        #         if x > m + 0.5:
        #             x = m + 0.5
        #         i0 = int(x)
        #         i1 = i0 + 1

        #         if y < 0.5:
        #             y = 0.5
        #         if y > n + 0.5:
        #             y = n + 0.5
        #         j0 = int(y)
        #         j1 = j0 + 1

        #         s1 = x - i0
        #         s0 = 1.0 - s1
        #         t1 = y - j0
        #         t0 = 1.0 - t1

        #         print(f"i={i}, j={j}, x={x}, y={y}, i0={i0}, i1={i1}, j0={j0}, j1={j1}")
        #         d[i, j] = s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) + s1 * (
        #             t0 * d0[i1, j0] + t1 * d0[i1, j1]
        #         )

        self.set_bnd(b, d)

    def run_simulation(self):
        grids = []
        for i in ap.alive_it(range(self.total_steps)):
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

    @property
    def m(self) -> int:
        return self.grid.m

    @property
    def n(self) -> int:
        return self.grid.n

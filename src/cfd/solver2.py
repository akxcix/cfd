import copy

import alive_progress as ap
import numba as nb
import numpy as np
from matplotlib import pyplot as plt

from .space import Space2d, SpacePoint


class Solver:
    def __init__(self, space: Space2d, nu: float, rho: float, dt: float, F: float):
        self.space = space

        self.nu = nu
        self.rho = rho
        self.dt = dt
        self.F = F

        # V stands for velocity.
        self.vx = np.array([([0] * space.nx) for _ in range(space.ny)]).astype("f")
        self.vy = np.array([([0] * space.nx) for _ in range(space.ny)]).astype("f")
        self.p = np.array([([0] * space.nx) for _ in range(space.ny)]).astype("f")

        # Initialize v and p with grids
        for i in range(space.ny):
            for j in range(space.nx):
                self.vx[i][j] = space.grid[i][j].v.vx
                self.vy[i][j] = space.grid[i][j].v.vy
                self.p[i][j] = space.grid[i][j].p
        # print(self.p)

        # self.obstacle = np.zeros([space.ny, space.nx]).astype("i")
        # for i in range(space.ny):
        #     for j in range(space.nx):
        #         if (i - int(space.ny / 2)) ** 2 + (j - int(space.nx / 2)) ** 2 < 50**2:
        #             self.obstacle[i, j] = 1
        self.obstacle = as_grid(
            x_min=-0.1,
            x_max=1.1,
            y_min=-3,
            y_max=3,
            x_points=space.nx,
            y_points=space.ny,
            params=[0, 0, 0, 0, 0],
        )[::-1]
        plt.imshow(self.obstacle)
        plt.show()

    def solve(self, iters):
        import time

        for iter in ap.alive_it(range(iters)):
            start = time.time()
            self.vx, self.vy, self.p = next_step(
                self.dt,
                self.vx,
                self.vy,
                self.p,
                self.space.dx,
                self.space.dy,
                self.space.nx,
                self.space.ny,
                self.nu,
                self.rho,
                self.F,
                self.obstacle,
            )
            end = time.time()
            print(f"iter={iter}, time={end-start}")
            if iter % 1000 == 0:
                for i in range(self.space.ny):
                    for j in range(self.space.nx):
                        self.space.grid[i][j].v.vx = self.vx[i][j]
                        self.space.grid[i][j].v.vy = self.vy[i][j]
                        self.space.grid[i][j].p = self.p[i][j]

                self.space.plot_pressure()
                self.space.plot_velocity(scale=1)


@nb.njit
def next_step(dt, vx, vy, p, dx, dy, nx, ny, nu, rho, F, obstacle):
    dtdx = dt / dx
    dtdy = dt / dy

    new_vx = vx.copy()
    new_vy = vy.copy()
    new_p = p.copy()

    for i in range(nx):
        for j in range(ny):
            if j == 0:
                new_vx[i][j] = 10
                new_vy[i][j] = 0
                new_p[i][j] = 1
                continue

            if j == nx - 1:
                new_vx[i][j] = new_vx[i - 1][j]
                new_vy[i][j] = new_vy[i - 1][j]
                new_p[i][j] = -1
                continue

            if i == 0:
                new_vx[i][j] = 0
                new_vy[i][j] = 0
                new_p[i][j] = new_p[i + 1][j]
                continue

            if i == ny - 1:
                new_vx[i][j] = 0
                new_vy[i][j] = 0
                new_p[i][j] = new_p[i - 1][j]
                continue

            # obstacle_radius = 5
            if obstacle[i][j]:
                new_vx[i][j] = new_vy[i][j] = new_p[i][j] = 0
                continue

            vx_p = (
                vx[i][j]
                - vx[i][j] * dtdx * (vx[i][j] - vx[i - 1][j])
                - vy[i][j] * dtdy * (vx[i][j] - vx[i][j - 1])
                - (dtdx / (2 * rho)) * (p[i + 1][j] - p[i - 1][j])
                + nu * ((dtdx / dx) * (vx[i + 1][j] + vx[i - 1][j] - 2 * vx[i][j]))
                + nu * ((dtdy / dy) * (vx[i][j + 1] + vx[i][j - 1] - 2 * vx[i][j]))
                + dt * F
            )

            vy_p = (
                vy[i][j]
                - vx[i][j] * dtdx * (vy[i][j] - vy[i - 1][j])
                - vy[i][j] * dtdy * (vy[i][j] - vy[i][j - 1])
                - (dtdy / (2 * rho)) * (p[i][j + 1] - p[i][j - 1])
                + nu * ((dtdx / dx) * (vy[i + 1][j] + vy[i - 1][j] - 2 * vy[i][j]))
                + nu * ((dtdy / dy) * (vy[i][j + 1] + vy[i][j - 1] - 2 * vy[i][j]))
            )

            p_p = (
                (
                    (p[i + 1][j] + p[i - 1][j]) * dy * dy
                    + (p[i][j + 1] + p[i][j - 1]) * dx * dx
                )
                / (2 * (dx**2 + dy**2))
            ) - calc_brackets(
                dx,
                dy,
                dt,
                rho,
                vx[i + 1][j],
                vx[i - 1][j],
                vx[i][j + 1],
                vx[i][j - 1],
                vy[i][j + 1],
                vy[i][j - 1],
                vy[i + 1][j],
                vy[i + 1][j],
            )

            new_vx[i][j] = vx_p
            new_vy[i][j] = vy_p
            new_p[i][j] = p_p

    return new_vx, new_vy, new_p


@nb.njit
def calc_brackets(
    dx,
    dy,
    dt,
    rho,
    vx_ip1_j,
    vx_im1_j,
    vx_i_jp1,
    vx_i_jm1,
    vy_i_jp1,
    vy_i_jm1,
    vy_ip1_j,
    vy_im1_j,
):
    # print(
    #     dx,
    #     dy,
    #     dt,
    #     rho,
    #     vx_ip1_j,
    #     vx_im1_j,
    #     vx_i_jp1,
    #     vx_i_jm1,
    #     vy_i_jp1,
    #     vy_i_jm1,
    #     vy_ip1_j,
    #     vy_im1_j,
    # )
    p1 = rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2))
    t1 = (1 / dt) * (
        ((vx_ip1_j - vx_im1_j) / (2 * dx)) + ((vy_i_jp1 - vy_i_jm1) / (2 * dy))
    )
    t2 = ((vx_ip1_j - vx_im1_j) / (2 * dx)) ** 2
    t3 = 2 * ((vx_i_jp1 - vx_i_jm1) / (2 * dy)) * ((vy_ip1_j - vy_im1_j) / (2 * dx))
    t4 = ((vy_i_jp1 - vy_i_jm1) / (2 * dy)) ** 2

    return p1 * (t1 - t2 - t3 - t4)


CS = 0.2969
C1 = -0.1260
C2 = -0.3516
C3 = 0.2843
C4 = -0.1015


def f(x, cs=CS, c1=C1, c2=C2, c3=C3, c4=C4):
    return cs * np.sqrt(x) + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4


def g(x, a, b, c, d, e):
    return x * (x - 1) * (a * x**0.5 + b + c * x + d * x**2 + e * x**3)


def g_(x, a, b, c, d, e, add):
    gv = g(x, a, b, c, d, e)
    fv = f(x) * 2

    if add:
        return gv + fv
    else:
        return gv - fv


def as_grid(
    *,
    x_min: float = 0,
    x_max: float = 1,
    y_min: float = -1,
    y_max: float = 1,
    x_points: int = 100,
    y_points: int = 1000,
    params: np.ndarray | list[float],
):
    x_linspace = np.linspace(x_min, x_max, x_points)
    y_linspace = np.linspace(y_min, y_max, y_points)

    # print(x_linspace)
    upper = g_(x_linspace, *params, True)
    lower = g_(x_linspace, *params, False)

    outputs = (
        np.ones([len(x_linspace), len(y_linspace)], dtype=int)
        & (y_linspace[np.newaxis, :] <= upper[:, np.newaxis])
        & (y_linspace[np.newaxis, :] >= lower[:, np.newaxis])
    ).astype("i")
    assert outputs.shape == (len(x_linspace), len(y_linspace))
    return outputs

from __future__ import annotations

import dataclasses as dcls

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from . import utils


@dcls.dataclass
class Velocity:
    vx = 0
    vy = 0


@dcls.dataclass
class GridPoint:
    """
    a particular point x,y on the grid
    """

    ux: float = 0
    uy: float = 0
    rho: float = 0
    p: float = 0
    t: float = 0

    def __add__(self, other: GridPoint | int | float) -> GridPoint:
        if isinstance(other, (int, float)):
            return GridPoint(
                self.ux + other,
                self.uy + other,
                self.rho + other,
                self.p + other,
                self.t + other,
            )
        return GridPoint(
            self.ux + other.ux,
            self.uy + other.uy,
            self.rho + other.rho,
            self.p + other.p,
            self.t + other.t,
        )

    def __radd__(self, other: GridPoint | int | float) -> GridPoint:
        return other + self

    def __mul__(self, k: int | float):
        if not isinstance(k, (int, float)):
            raise TypeError("The multiplier must be a number.")

        return GridPoint(self.ux * k, self.uy * k, self.rho * k, self.p * k, self.t * k)

    def __rmul__(self, k: int | float):
        return self.__mul__(k)

    def __truediv__(self, k: int | float):
        return self * (1 / k)


class Grid:
    def __init__(self, m: int, n: int, init: bool = False) -> None:
        self.m = m
        self.n = n
        self.grid = [GridPoint() for _ in range(m * n)]

        mu_y = n // 2
        mu_x = m // 2
        sigma_y = m // 4
        sigma_x = n // 4

        if init:
            for i in range(m):
                for j in range(n):
                    idx = self._index(i, j)

                    rho = utils.gaussian(i, j, mu_x, mu_y, sigma_x, sigma_y)
                    p = 101325
                    ux = np.random.rand() * 7 * utils.random_sign()
                    uy = np.random.rand() * 7 * utils.random_sign()
                    # ux = 1
                    # uy = 1

                    self.grid[idx].rho = rho
                    self.grid[idx].p = p
                    self.grid[idx].ux = ux
                    self.grid[idx].uy = uy

    def __getitem__(self, key: tuple[int, int]) -> GridPoint:
        x, y = key
        return self._get_at(x, y)

    def __setitem__(self, key: tuple[int, int], newpoint: GridPoint):
        x, y = key
        self._set_at(x, y, newpoint)

    def _get_at(self, x: int, y: int):
        index = self._index(x, y)
        return self.grid[index]

    def _set_at(self, x: int, y: int, newpoint: GridPoint):
        index = self._index(x, y)
        self.grid[index] = newpoint

    def _index(self, x: int, y: int) -> int:
        return y * self.m + x

    def show_grid(
        self, min_rho, max_rho, plot_rho=True, plot_p=False, plot_velocity_field=False
    ):
        rho_values = np.zeros([self.m, self.n])
        ux_values = np.zeros([self.m, self.n])
        uy_values = np.zeros([self.m, self.n])

        for i in range(self.m):
            for j in range(self.n):
                rho_values[i, j] = self.grid[self._index(i, j)].rho
                ux_values[i, j] = self.grid[self._index(i, j)].ux
                uy_values[i, j] = self.grid[self._index(i, j)].uy

        if plot_rho:
            self.plot_scalar_field(rho_values, "Density (rho)", min_rho, max_rho)

        # if plot_p:
        #     p_values = np.array([point.p for point in self.grid]).reshape((self.m, self.n))
        #     self.plot_scalar_field(p_values, 'Pressure (p)')

        if plot_velocity_field:
            self.plot_velocity_field(ux_values, uy_values)

    def plot_scalar_field(self, values, title, vmin, vmax):
        plt.figure(figsize=(6, 5))
        im = plt.imshow(values, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, label="Value")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.show()

    def plot_velocity_field(self, ux_values, uy_values):
        Y, X = np.mgrid[0 : self.m, 0 : self.n]
        plt.figure(figsize=(6, 5))
        plt.quiver(X, Y, ux_values, uy_values)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Velocity Field")
        plt.xlim(-0.5, self.n - 0.5)
        plt.ylim(-0.5, self.m - 0.5)
        plt.show()

    @staticmethod
    def plot_grids(output_file, grid_list, title="Grid Evolution", vmin=None, vmax=None):
        fig, ax = plt.subplots()

        initial_grid = grid_list[0]
        rho_values = np.zeros([initial_grid.m, initial_grid.n])
        for x in range(initial_grid.m):
            for y in range(initial_grid.n):
                rho_values[x, y] = initial_grid[x, y].rho

        im = ax.imshow(rho_values, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label="Density (rho)")

        def init():
            return (im,)

        def update(frame):
            current_grid = grid_list[frame]

            rho_values = np.zeros([current_grid.m, current_grid.n])
            for x in range(current_grid.m):
                for y in range(current_grid.n):
                    rho_values[x, y] = current_grid[x, y].rho

            im.set_data(rho_values)
            ax.set_title(f"Time Step: {frame}")
            return (im,)

        ani = animation.FuncAnimation(
            fig, update, frames=len(grid_list), init_func=init, blit=True
        )
        ani.save(output_file)

        plt.close(fig)

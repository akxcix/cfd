from __future__ import annotations

import dataclasses as dcls

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .utils import Utils


@dcls.dataclass(frozen=True)
class GridPoint:
    """
    a particular point x,y on the grid
    """

    ux: float
    uy: float
    rho: float
    p: float
    t: float

    def __add__(self, other) -> GridPoint:
        return GridPoint(
            self.ux + other.ux,
            self.uy + other.uy,
            self.rho + other.rho,
            self.p + other.p,
            self.t + other.t,
        )

    def __mul__(self, k: int | float):
        if not isinstance(k, (int, float)):
            raise TypeError("The multiplier must be a number.")

        return GridPoint(self.ux * k, self.uy * k, self.rho * k, self.p * k, self.t * k)

    def __rmul__(self, k: int | float):
        return self.__mul__(k)


class Grid:
    def __init__(self, m, n, init=False) -> None:
        self.m = m
        self.n = n
        self.grid = [GridPoint() for _ in range(m * n)]

        mu_y = m // 2
        mu_x = n // 2
        sigma_y = m // 4
        sigma_x = n // 4

        if init:
            for i in range(m * n):
                y = i // n
                x = i % n

                rho = Utils.gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y)
                p = 101325
                ux = np.random.rand() * Utils.random_sign()
                uy = np.random.rand() * Utils.random_sign()

                self.grid[i].rho = rho
                self.grid[i].p = p
                self.grid[i].ux = ux
                self.grid[i].uy = uy

    def get_at(self, x, y):
        index = y * self.n + x
        return self.grid[index]

    def set_at(self, x, y, newpoint):
        index = y * self.n + x
        self.grid[index] = newpoint

    def show_grid(self):
        rho_values = np.array([point.rho for point in self.grid]).reshape(
            (self.m, self.n)
        )
        p = np.array([point.p for point in self.grid]).reshape((self.m, self.n))
        ux = np.array([point.ux for point in self.grid]).reshape((self.m, self.n))
        uy = np.array([point.uy for point in self.grid]).reshape((self.m, self.n))

        self.plot(rho_values)
        # self.plot(p)
        # self.plot(ux)
        # self.plot(uy)

    def animate_2d(solutions):
        fig, ax = plt.subplots(figsize=(6, 5))

        def update_plot(frame_number):
            ax.clear()
            contour = ax.contourf(
                X, Y, solutions[frame_number], levels=50, cmap="viridis"
            )
            return (contour,)

        ani = animation.FuncAnimation(
            fig, update_plot, frames=len(solutions), blit=True
        )
        plt.close(fig)
        return ani

    def plot(self, values):
        plt.imshow(values, origin="lower", cmap="viridis")
        plt.colorbar(label="Value")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

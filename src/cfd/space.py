from __future__ import annotations
from dataclasses import dataclass as dcls
import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------------------------------------------

@dcls
class Velocity:
    def __init__(
        self, 
        vx: float, 
        vy: float,
    ):
        self.vx = vx
        self.vy = vy

    def __add__(self, other: Velocity):
        vx = self.vx + other.vx
        vy = self.vy + other.vy

        return Velocity(vx, vy)

@dcls
class SpacePoint:
    def __init__(
        self,
        v: Velocity = Velocity(0,0),
        rho: float = 0.1,
        p: float = 0,
        is_blocked: bool = False,
    ) -> None:
        """
        args:
            v: velocity
            rho: density
            p: pressure
            is_blocked: is this point free to contain fluid or blocked
        """
        self.is_blocked = False
        self.v = v
        self.rho = rho
        self.p = p
        self.is_blocked = is_blocked

@dcls
class Space2d:
    def __init__(
        self,
        lx: float,
        ly: float,
        nx: int,
        ny: int,
    ) -> None:
        """
        args:
            nx: size of grid in x direction
            ny: size of grid in y direction
        """
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny

        self.dx = lx/nx
        self.dy = ly/ny

        self.grid = [[SpacePoint() for px in range(nx)] for py in range(ny)]

    
    # representations ----------------------------------------------------------
    def plot_velocity(
        self, 
        scale: float = 1,
    ) -> None:
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        u = np.zeros((self.ny, self.nx))
        v = np.zeros((self.ny, self.nx))

        for i in range(self.ny):
            for j in range(self.nx):
                if not self.grid[i][j].is_blocked:
                    u[i, j] = self.grid[i][j].v.vx
                    v[i, j] = self.grid[i][j].v.vy

        fig, ax = plt.subplots()
        ax.quiver(x, y, u, v, scale=scale, scale_units='xy')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.set_xticks(np.arange(self.nx))
        ax.set_yticks(np.arange(self.ny))

        ax.set_title('Velocity Field')

        plt.show()

    def plot_density(self) -> None:
        density_grid = np.zeros((self.ny, self.nx))
        for i in range(self.ny):
            for j in range(self.nx):
                if not self.grid[i][j].is_blocked:
                    density_grid[i][j] = self.grid[i][j].rho

        fig, ax = plt.subplots()
        im = ax.imshow(density_grid, cmap='viridis', origin='lower')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.set_xticks(np.arange(self.nx))
        ax.set_yticks(np.arange(self.ny))

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Density', rotation=-90, va='bottom')

        ax.set_title('Density Grid')

        plt.show()

    def plot_pressure(self) -> None:
        pressure_grid = np.zeros((self.ny, self.nx))
        for i in range(self.ny):
            for j in range(self.nx):
                if not self.grid[i][j].is_blocked:
                    pressure_grid[i][j] = self.grid[i][j].p

        fig, ax = plt.subplots()
        im = ax.imshow(pressure_grid, cmap='viridis', origin='lower',vmax=10, vmin=-10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.set_xticks(np.arange(self.nx))
        ax.set_yticks(np.arange(self.ny))

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Pressure', rotation=-90, va='bottom')

        ax.set_title('Pressure Grid')

        plt.show()

    def __str__(self) -> str:
        """
        generated via claude (https://claude.ai)
        """
        def format_value(value: float) -> str:
            return f"{value:6.2f}"

        output = "Velocity X:\n"
        for row in self.grid:
            output += " ".join(
                "   X  " if point.is_blocked else format_value(point.v.vx)
                for point in row
            )
            output += "\n"

        output += "\nVelocity Y:\n"
        for row in self.grid:
            output += " ".join(
                "   X  " if point.is_blocked else format_value(point.v.vy)
                for point in row
            )
            output += "\n"

        output += "\nDensity:\n"
        for row in self.grid:
            output += " ".join(
                "   X  " if point.is_blocked else format_value(point.rho)
                for point in row
            )
            output += "\n"

        output += "\nPressure:\n"
        for row in self.grid:
            output += " ".join(
                "   X  " if point.is_blocked else format_value(point.p)
                for point in row
            )
            output += "\n"

        output += "\nBlocked:\n"
        for row in self.grid:
            output += " ".join(
                "   X  " if point.is_blocked else "   .  "
                for point in row
            )
            output += "\n"

        return output

import numpy as np
import matplotlib.pyplot as plt
from .utils import Utils


class GridPoint:
    """
    a particular point x,y on the grid
    """
    def __init__(self, ux=0, uy=0, rho=0, p=0, t=0) -> None:
        self.ux = ux
        self.uy = uy
        self.rho = rho
        self.p = p
        self.t = t

    def __add__(self, other):
        return GridPoint(self.ux + other.ux, self.uy + other.uy, 
                         self.rho + other.rho, self.p + other.p, 
                         self.t + other.t)
    
    def __mul__(self, k):
        if not isinstance(k, (int, float)):
            raise TypeError("The multiplier must be a number.")
        return GridPoint(self.ux * k, self.uy * k, 
                         self.rho * k, self.p * k, 
                         self.t * k)

    def __rmul__(self, k):
        return self.__mul__(k)

class Grid:
    def __init__(self, m, n, init=False) -> None:
        self.m = m
        self.n = n
        self.grid = [GridPoint() for _ in range(m*n)]

        mu_y = m//2
        mu_x = n //2
        sigma_y = m//4
        sigma_x = n//4

        if init:
            for i in range(m*n):
                y = i // n
                x = i % n

                rho = Utils.gaussian(x, y, mu_x, mu_y, sigma_x,sigma_y)
                p = 101325
                ux = 2
                uy = 0*ux//2

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
        rho_values = np.array([point.rho for point in self.grid]).reshape((self.m, self.n))
        p = np.array([point.p for point in self.grid]).reshape((self.m, self.n))
        ux = np.array([point.ux for point in self.grid]).reshape((self.m, self.n))
        uy = np.array([point.uy for point in self.grid]).reshape((self.m, self.n))

        self.plot(rho_values)
        # self.plot(p)
        # self.plot(ux)
        # self.plot(uy)

    def plot(self, values):
        plt.imshow(values, origin='lower', cmap='viridis')
        plt.colorbar(label='Value')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,y, mu_x, mu_y, sig_x, sig_y):
    a = (x-mu_x)**2
    b = (y-mu_y)**2
    c = 2*sig_x**2
    d = 2*sig_y**2

    e = a/c
    f = b/d

    g = -1 * (e + f)

    return np.exp(g)


class Grid:
    def __init__(self, shape: list[int]) -> None:
        self.grid = np.zeros(shape)
        m,n = shape
        mu_y = m//2
        mu_x = n//2
        sigma_y = m//4
        sigma_x = n//4
        a = 100
        for y in range(m):
            for x in range(n):
                self.grid[y, x] = a * gaussian(x, y, mu_x, mu_y, sigma_x,sigma_y)


    def print_grid(self):
        print(self.grid)

    def show_grid(self):
        plt.imshow(self.grid, origin='lower', cmap='viridis')
        plt.colorbar(label='Value')
        # plt.title('2D Gaussian Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

import numpy as np


def gaussian(x, y, mu_x, mu_y, sig_x, sig_y):
    a = (x - mu_x) ** 2
    b = (y - mu_y) ** 2
    c = 2 * sig_x**2
    d = 2 * sig_y**2

    e = a / c
    f = b / d

    g = -1 * (e + f)

    return np.exp(g)


def random_sign():
    r = np.random.rand()
    if r < 0.5:
        return -1
    return 1

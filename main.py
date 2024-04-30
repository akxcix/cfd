import numpy as np

import cfd

if __name__ == "__main__":
    x = np.arange(0, 1, 0.1)
    y = np.arange(0, 1, 0.1)

    v_x, v_y = np.meshgrid(x, y)
    v_x = np.sin(v_x)
    v_y = np.cos(v_y)

    cfd.quiver(x, y, v_x, v_y)

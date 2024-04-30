from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def quiver(
    x: NDArray, y: NDArray, v_x: NDArray, v_y: NDArray, fname: str | Path = ""
) -> None:
    plt.clf()

    assert (
        x.ndim == y.ndim == 1
    ), f"x and y must be 1D arrays. Got {x.ndim} and {y.ndim}."

    assert v_x.shape == v_y.shape == (len(x), len(y)), (
        "Velocity field and meshgrid shape mismatch. "
        f"Got {v_x.shape} and {v_y.shape}. Expected {(len(x), len(y))}"
    )

    _, ax = plt.subplots()
    q = ax.quiver(x, y, v_x, v_y)
    ax.quiverkey(q, X=0.3, Y=1.1, U=10, label="Quiver key, length = 10", labelpos="E")

    if fname:
        plt.savefig(fname)
    else:
        plt.show()

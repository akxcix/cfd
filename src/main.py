import abc
from typing import Protocol, Tuple

import alive_progress as ap
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from ax.service.ax_client import AxClient, ObjectiveProperties
from cmaes import CMA
from numpy.typing import NDArray

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# Global constants
NX: int = 1000  # resolution x-dir
NY: int = 200  # resolution y-dir
NT: int = 4000  # number of timesteps
RHO_0: float = 100.0  # average density
TAU: float = 0.6  # collision timescale
PLOT_REAL_TIME: bool = False  # switch on for plotting as the simulation goes along

# Airfoil constants
CS: float = 0.2969
C1: float = -0.1260
C2: float = -0.3516
C3: float = 0.2843
C4: float = -0.1015

PARAMS = [0.1, 0.2, 0.1, 0.1, 0.2]


def update_forces(
    F: torch.Tensor, cxs: torch.Tensor, cys: torch.Tensor, cylinder: torch.Tensor
) -> Tuple[float, float]:
    original_momentum_x: torch.Tensor = torch.einsum("ijk,k->ij", F, cxs.float())
    original_momentum_y: torch.Tensor = torch.einsum("ijk,k->ij", F, cys.float())
    delta_momentum_x: torch.Tensor = -2 * original_momentum_x
    delta_momentum_y: torch.Tensor = -2 * original_momentum_y
    drag_force: float = (delta_momentum_x * cylinder).sum().item()
    lift_force: float = (delta_momentum_y * cylinder).sum().item()
    return drag_force, lift_force


def f(
    x: NDArray,
    cs: float = CS,
    c1: float = C1,
    c2: float = C2,
    c3: float = C3,
    c4: float = C4,
) -> NDArray:
    return cs * np.sqrt(x) + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4


def g(x: NDArray, a: float, b: float, c: float, d: float, e: float) -> NDArray:
    return x * (x - 1) * (a * x**0.5 + b + c * x + d * x**2 + e * x**3)


def g_(
    x: NDArray, a: float, b: float, c: float, d: float, e: float, add: bool
) -> NDArray:
    gv: NDArray = g(x, a, b, c, d, e)
    fv: NDArray = f(x)
    return gv + fv if add else gv - fv


def as_grid(
    x_min: float = -0.1,
    x_max: float = 1.5,
    y_min: float = -0.5,
    y_max: float = 0.5,
    x_points: int = NX,
    y_points: int = NY,
    params: NDArray | list[float] = PARAMS,
) -> NDArray:
    x_linspace: NDArray = np.linspace(x_min, x_max, x_points)
    y_linspace: NDArray = np.linspace(y_min, y_max, y_points)
    upper: NDArray = g_(x_linspace, *params, True)
    lower: NDArray = g_(x_linspace, *params, False)
    outputs: NDArray = (
        np.ones([len(x_linspace), len(y_linspace)], dtype=int)
        & (y_linspace[np.newaxis, :] <= upper[:, np.newaxis])
        & (y_linspace[np.newaxis, :] >= lower[:, np.newaxis])
    ).astype("i")
    assert outputs.shape == (len(x_linspace), len(y_linspace))
    return outputs.T


@torch.no_grad()
def run(params: NDArray) -> None:

    # Simulation parameters
    NL: int = 9
    idxs: torch.Tensor = torch.arange(NL)
    cxs: torch.Tensor = torch.tensor([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys: torch.Tensor = torch.tensor([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights: torch.Tensor = torch.tensor(
        [4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36]
    )

    F: torch.Tensor = torch.ones((NY, NX, NL))
    np.random.seed(42)
    F += 0.01 * torch.randn(NY, NX, NL)
    X, Y = np.meshgrid(range(NX), range(NY))
    X, Y = torch.tensor(X), torch.tensor(Y)
    F[:, :, 3] += 2 * (1 + 0.2 * torch.cos(2 * np.pi * X / NX * 4))
    rho: torch.Tensor = torch.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= RHO_0 / rho

    # Cylinder boundary
    cylinder: torch.Tensor = torch.tensor(as_grid(params=params)).bool()

    # Prep figure
    fig = plt.figure(figsize=(4, 2), dpi=80)

    F, cxs, cys, cylinder = (
        F.to(DEVICE),
        cxs.to(DEVICE),
        cys.to(DEVICE),
        cylinder.to(DEVICE),
    )

    lift_drag_ratio = []
    # Simulation Main Loop
    for it in range(NT):
        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = torch.roll(F[:, :, i], cx.item(), 1)
            F[:, :, i] = torch.roll(F[:, :, i], cy.item(), 0)

        # Set reflective boundaries
        bndryF: torch.Tensor = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Calculate fluid variables
        rho = torch.sum(F, 2)
        ux: torch.Tensor = torch.sum(F * cxs, 2) / rho
        uy: torch.Tensor = torch.sum(F * cys, 2) / rho

        # Apply Collision
        Feq: torch.Tensor = torch.zeros_like(F)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = (
                rho
                * w
                * (
                    1
                    + 3 * (cx * ux + cy * uy)
                    + 9 * (cx * ux + cy * uy) ** 2 / 2
                    - 3 * (ux**2 + uy**2) / 2
                )
            )

        F += -(1.0 / TAU) * (F - Feq)

        # Apply boundary
        F[cylinder, :] = bndryF
        drag, lift = update_forces(F, cxs, cys, cylinder)
        lift_drag_ratio.append(lift / (drag + 1e-8))
        print(f"Drag: {drag}, Lift: {lift}")

        # Real-time plotting
        if PLOT_REAL_TIME and (it % 10 == 0 or it == NT - 1):
            plot_flow(
                cylinder.clone().cpu().numpy(),
                ux.clone().cpu().numpy(),
                uy.clone().cpu().numpy(),
                fig,
            )
        return np.mean(sorted(lift_drag_ratio)[1:-1])

    # Save figure
    if PLOT_REAL_TIME:
        plt.savefig("latticeboltzmann.png", dpi=240)
        plt.show()


def plot_flow(cylinder: NDArray, ux: NDArray, uy: NDArray, fig: plt.Figure):
    plt.cla()
    ux[cylinder] = 0
    uy[cylinder] = 0
    vorticity = (np.roll(ux, -1, 0) - np.roll(ux, 1, 0)) - (
        np.roll(uy, -1, 1) - np.roll(uy, 1, 1)
    )
    vorticity[cylinder] = np.nan
    vorticity = np.ma.array(vorticity, mask=cylinder)
    plt.imshow(vorticity, cmap="bwr")
    plt.imshow(~cylinder, cmap="gray", alpha=0.3)
    plt.clim(-0.1, 0.1)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")
    plt.pause(0.001)


class Optimizer(Protocol):
    @abc.abstractmethod
    def ask(self) -> NDArray: ...

    @abc.abstractmethod
    def tell(self, solutions: list[Tuple[NDArray, float]]) -> None: ...

    @abc.abstractmethod
    def total(self) -> int: ...


class CMAOptimizer(Optimizer):
    def __init__(self, population_size: int = 10, loops: int = 100) -> None:
        super().__init__()

        self._loop = loops
        self._cma = CMA(mean=np.zeros(5), sigma=0.2, population_size=population_size)

    def ask(self) -> NDArray:
        return self._cma.ask()

    def tell(self, solutions: Tuple[NDArray, float]) -> None:
        self._cma.tell([solutions])

    def total(self) -> int:
        return self._loop * self._cma.population_size


LIFT_DRAG_RATIO = "lift_drag_ratio"


class AxOptimizer(Optimizer):
    def __init__(self, total: int) -> None:
        super().__init__()

        self._total = total
        self._ax_client = AxClient()
        self._ax_client.create_experiment(
            name="bayesian",
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-0.5, 0.5],
                    "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                },
                {
                    "name": "x2",
                    "type": "range",
                    "bounds": [-0.5, 0.5],
                },
                {
                    "name": "x3",
                    "type": "range",
                    "bounds": [-0.5, 0.5],
                },
                {
                    "name": "x4",
                    "type": "range",
                    "bounds": [-0.5, 0.5],
                },
                {
                    "name": "x5",
                    "type": "range",
                    "bounds": [-0.5, 0.5],
                },
            ],
            objectives={LIFT_DRAG_RATIO: ObjectiveProperties(minimize=False)},
            parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
            outcome_constraints=["l2norm <= 1.25"],  # Optional.
        )
        self._last_trial = -1
        self._last_x = None

    def ask(self) -> NDArray:
        self._last_x, self._last_trial = self._ax_client.get_next_trial()
        return self._last_x

    def tell(self, solutions: Tuple[NDArray, float]) -> None:
        # HACK:
        # This allows Ax to use CMA API.
        if solutions[0] is not self._last_x:
            raise ValueError("The last x is not the same as the x provided")

        self._ax_client.complete_trial(
            trial_index=self._last_trial, raw_data={LIFT_DRAG_RATIO: solutions[1]}
        )

    def total(self) -> int:
        return self._total

def run(algo):
    if algo == 'cma':
        optimizer = CMAOptimizer()
    elif algo == 'ax':
        optimizer = AxOptimizer()
    else:
        raise ValueError(f"Unknown algorithm {algo}")

    for generation in ap.alive_it(optimizer.total()):
        solutions = []
        x = optimizer.ask()
        value = run(x)
        solutions.append((x, value))
        print(f"#{generation} {value} (x={x})")
        optimizer.tell(solutions)

if __name__ == "__main__":


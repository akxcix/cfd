import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple

# Global constants
Nx: int = 1000  # resolution x-dir
Ny: int = 200   # resolution y-dir
Nt: int = 4000  # number of timesteps
rho0: float = 100.0  # average density
tau: float = 0.6  # collision timescale
plotRealTime: bool = True  # switch on for plotting as the simulation goes along

# Airfoil constants
CS: float = 0.2969
C1: float = -0.1260
C2: float = -0.3516
C3: float = 0.2843
C4: float = -0.1015

PARAMS = [.1, .2, .1, .1, 0.2]


def update_forces(F: torch.Tensor, cxs: torch.Tensor, cys: torch.Tensor, cylinder: torch.Tensor) -> Tuple[float, float]:
    original_momentum_x: torch.Tensor = torch.einsum("ijk,k->ij", F, cxs.float())
    original_momentum_y: torch.Tensor = torch.einsum("ijk,k->ij", F, cys.float())
    delta_momentum_x: torch.Tensor = -2 * original_momentum_x
    delta_momentum_y: torch.Tensor = -2 * original_momentum_y
    drag_force: float = (delta_momentum_x * cylinder).sum().item()
    lift_force: float = (delta_momentum_y * cylinder).sum().item()
    return drag_force, lift_force


def f(x: np.ndarray, cs: float = CS, c1: float = C1, c2: float = C2, c3: float = C3, c4: float = C4) -> np.ndarray:
    return cs * np.sqrt(x) + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4


def g(x: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    return x * (x - 1) * (a * x**0.5 + b + c * x + d * x**2 + e * x**3)


def g_(x: np.ndarray, a: float, b: float, c: float, d: float, e: float, add: bool) -> np.ndarray:
    gv: np.ndarray = g(x, a, b, c, d, e)
    fv: np.ndarray = f(x)
    return gv + fv if add else gv - fv


def as_grid(x_min: float = -0.1, x_max: float = 1.5, y_min: float = -0.5, y_max: float = 0.5, x_points: int = Nx,
            y_points: int = Ny, params: np.ndarray | list[float] = PARAMS) -> np.ndarray:
    x_linspace: np.ndarray = np.linspace(x_min, x_max, x_points)
    y_linspace: np.ndarray = np.linspace(y_min, y_max, y_points)
    upper: np.ndarray = g_(x_linspace, *params, True)
    lower: np.ndarray = g_(x_linspace, *params, False)
    outputs: np.ndarray = (
        np.ones([len(x_linspace), len(y_linspace)], dtype=int)
        & (y_linspace[np.newaxis, :] <= upper[:, np.newaxis])
        & (y_linspace[np.newaxis, :] >= lower[:, np.newaxis])
    ).astype("i")
    assert outputs.shape == (len(x_linspace), len(y_linspace))
    return outputs.T


def main() -> int:
    torch.set_grad_enabled(False)
    DEVICE: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(DEVICE)

    # Simulation parameters
    NL: int = 9
    idxs: torch.Tensor = torch.arange(NL)
    cxs: torch.Tensor = torch.tensor([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys: torch.Tensor = torch.tensor([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights: torch.Tensor = torch.tensor([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

    F: torch.Tensor = torch.ones((Ny, Nx, NL))
    np.random.seed(42)
    F += 0.01 * torch.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    X, Y = torch.tensor(X), torch.tensor(Y)
    F[:, :, 3] += 2 * (1 + 0.2 * torch.cos(2 * np.pi * X / Nx * 4))
    rho: torch.Tensor = torch.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # Cylinder boundary
    cylinder: torch.Tensor = torch.tensor(as_grid(params=PARAMS)).bool()

    # Prep figure
    fig = plt.figure(figsize=(4, 2), dpi=80)

    F, cxs, cys, cylinder = F.to(DEVICE), cxs.to(DEVICE), cys.to(DEVICE), cylinder.to(DEVICE)

    # Simulation Main Loop
    for it in tqdm(range(Nt)):
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
            Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) + 9 * (cx * ux + cy * uy) ** 2 / 2 - 3 * (ux**2 + uy**2) / 2)

        F += -(1.0 / tau) * (F - Feq)

        # Apply boundary
        F[cylinder, :] = bndryF
        drag, lift = update_forces(F, cxs, cys, cylinder)
        print(f"Drag: {drag}, Lift: {lift}")

        # Real-time plotting
        if plotRealTime and (it % 10 == 0 or it == Nt - 1):
            plot_flow(cylinder.clone().cpu().numpy(), ux.clone().cpu().numpy(), uy.clone().cpu().numpy(), fig)

    # Save figure
    plt.savefig("latticeboltzmann.png", dpi=240)
    plt.show()

    return 0


def plot_flow(cylinder: np.ndarray, ux: np.ndarray, uy: np.ndarray, fig: plt.Figure):
    plt.cla()
    ux[cylinder] = 0
    uy[cylinder] = 0
    vorticity = (np.roll(ux, -1, 0) - np.roll(ux, 1, 0)) - (np.roll(uy, -1, 1) - np.roll(uy, 1, 1))
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


if __name__ == "__main__":
    main()

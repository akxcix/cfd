import argparse
import numpy as np

import cfd
# from cfd.grids import Grid
from cfd.space import Space2d, Velocity
from cfd.solver2 import Solver

def initialize_wind_tunnel(space: Space2d, inlet_velocity: float, outlet_pressure: float) -> None:
    for i in range(space.ny):
        space.grid[i][0].v = Velocity(inlet_velocity, 0)
        space.grid[i][0].rho = 1.0
        space.grid[i][0].p = 1.0

    for i in range(space.ny):
        space.grid[i][space.nx - 1].p = outlet_pressure

    for j in range(space.nx):
        space.grid[0][j].is_blocked = True
        space.grid[space.ny - 1][j].is_blocked = True

def initialize_ball(space: Space2d, ball_radius: float) -> None:
    center_x = space.nx // 2
    center_y = space.ny // 2

    for i in range(space.ny):
        for j in range(space.nx):
            distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if distance <= ball_radius:
                space.grid[i][j].is_blocked = True

def main():
    parser = argparse.ArgumentParser(description="CFD Simulation")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--plot", type=int, default=1, help="Plot interval")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--output_file", type=str, default="animation.mp4", help="Output file name")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[10, 10], help="Grid size (e.g., 10 10)")

    args = parser.parse_args()
    

    # grid = cfd.Grid(args.grid_size[0], args.grid_size[1], init=True)
    # solver = cfd.Solver(args.dt, args.iterations, grid, plot_interval=args.plot)
    # grid_list = solver.run_simulation()
    # Grid.plot_grids(args.output_file, grid_list, title="Grid Evolution", vmin=0, vmax=1)
    
    space = Space2d(10, 10, 20, 20)
    # initialize_wind_tunnel(space, inlet_velocity=1.0, outlet_pressure=0.0)

    ball_radius = 50.0
    # initialize_ball(space, ball_radius)

    # space.plot_velocity()
    space.plot_pressure()
    # space.plot_density()

    solver = Solver(space, 0.1,1,0.1)
    solver.solve(1000)


if __name__ == "__main__":
    main()
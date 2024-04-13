import argparse

import cfd
from cfd.grids import Grid


def main():
    parser = argparse.ArgumentParser(description="CFD Simulation")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--plot", type=int, default=1, help="Plot interval")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations"
    )
    parser.add_argument(
        "--output_file", type=str, default="animation.mp4", help="Output file name"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        default=[100, 100],
        help="Grid size (e.g., 10 10)",
    )

    args = parser.parse_args()

    grid = cfd.Grid(args.grid_size[0], args.grid_size[1], init=True)
    solver = cfd.Solver(args.dt, args.iterations, grid, plot_interval=args.plot)
    grid_list = solver.run_simulation()
    Grid.plot_grids(args.output_file, grid_list, title="Grid Evolution", vmin=0, vmax=1)


if __name__ == "__main__":
    main()

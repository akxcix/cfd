import fire

import cfd


def main(dt=0.1):
    grid = cfd.Grid(200, 500, init=True)
    # grid.show_grid()
    solver = cfd.Solver(dt, 500, grid)
    solver.run_simulation()


if __name__ == "__main__":
    fire.Fire(main)

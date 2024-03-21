import cfd

def main():
    grid = cfd.Grid(10,20, init=True)
    # grid.show_grid()
    solver = cfd.Solver(0.01, 1000, grid)
    solver.run_simulation()

if __name__ == "__main__":
    main()



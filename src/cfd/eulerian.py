from .grids import Grid
import numpy as np

class Eulerian:
    def __init__(self, dt, total_steps, grid) -> None:
        self.total_steps = total_steps
        self.dt = dt
        self.grid = grid
        self.gamma = 1.4

    def derivative(self, i, j, attribute):
        """Calculate the derivative using central differences."""
        attribute_grid = np.array([getattr(point, attribute) for point in self.grid.grid]).reshape(self.grid.m, self.grid.n)

        # Derivative in x-direction
        if j > 0 and j < self.grid.n - 1:
            d_attribute_dx = (attribute_grid[i, j + 1] - attribute_grid[i, j - 1]) / 2
        else:
            d_attribute_dx = 0  # Simple handling for boundaries

        # Derivative in y-direction
        if i > 0 and i < self.grid.m - 1:
            d_attribute_dy = (attribute_grid[i + 1, j] - attribute_grid[i - 1, j]) / 2
        else:
            d_attribute_dy = 0  # Simple handling for boundaries

        return d_attribute_dx, d_attribute_dy


    def step_once(self):
        m, n = self.grid.m, self.grid.n
        new_grid = Grid(m, n)

        for i in range(self.grid.m):
            for j in range(self.grid.n):
                index = i * self.grid.n + j
                current = self.grid.grid[index]
                
                # Compute derivatives for rho, ux, uy, and p
                drho_dx, drho_dy = self.derivative(i, j, 'rho')
                dux_dx, dux_dy = self.derivative(i, j, 'ux')
                duy_dx, duy_dy = self.derivative(i, j, 'uy')
                dp_dx, dp_dy = self.derivative(i, j, 'p')

                # Update equations - simplified, illustrative calculations
                new_rho = current.rho - self.dt * (current.ux * drho_dx + current.rho * dux_dx + current.uy * drho_dy + current.rho * duy_dy)
                new_ux = current.ux - self.dt * (current.ux * dux_dx + dux_dy * current.uy + (1 / current.rho) * dp_dx)
                new_uy = current.uy - self.dt * (current.uy * duy_dy + duy_dx * current.ux + (1 / current.rho) * dp_dy)
                new_p = current.p - self.dt * (current.ux * dp_dx + current.p * dux_dx + current.uy * dp_dy + current.p * duy_dy)

                # Update the new grid point
                new_grid.grid[index].rho = new_rho
                new_grid.grid[index].ux = new_ux
                new_grid.grid[index].uy = new_uy
                new_grid.grid[index].p = new_p

        # for i in range(m * n):
        #     # Extract grid point
        #     point = self.grid.grid[i]
        #     rho, ux, uy, p = point.rho, point.ux, point.uy, point.p

        #     # Placeholder for real calculations
        #     # Actual fluid dynamics computations would go here
        #     # For demonstration, simple updates with placeholders
        #     drho_dt = -rho * (ux + uy) * self.dt  # Placeholder
        #     dux_dt = -ux * self.dt  # Placeholder
        #     duy_dt = -uy * self.dt  # Placeholder
        #     dp_dt = -p * self.dt  # Placeholder

        #     # Assign new values to the grid point in the new grid
        #     new_grid.grid[i].rho = rho + drho_dt
        #     new_grid.grid[i].ux = ux + dux_dt
        #     new_grid.grid[i].uy = uy + duy_dt
        #     new_grid.grid[i].p = p + dp_dt
        #     # If you calculate temperature, update it similarly

        # Update the grid with the new values
        self.grid = new_grid

    def run_simulation(self):
        for i in range(self.total_steps):
            self.step_once()
            if i % 1 == 0:
                self.grid.show_grid()
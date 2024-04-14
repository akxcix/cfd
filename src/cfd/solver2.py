import numpy as np
from .space import Space2d, SpacePoint
import copy

class Solver:
    def __init__(self, space: Space2d, nu: float, rho: float, dt: float):
        self.space = space
        self.next_space = copy.deepcopy(space)

        self.nu = nu
        self.rho = rho
        self.dt = dt

    def calc_brackets(
            self,
            dx,
            dy,
            dt,
            rho,
            vx_ip1_j, 
            vx_im1_j,
            vx_i_jp1,
            vx_i_jm1,
            vy_i_jp1, 
            vy_i_jm1,
            vy_ip1_j,
            vy_im1_j,
        ):
        p1 = rho * dx**2 * dy**2 / (2*(dx**2 + dy**2))
        t1 = (1/dt) * (((vx_ip1_j - vx_im1_j)/(2*dx)) + ((vy_i_jp1 - vy_i_jm1)/(2*dy)))
        t2 = ((vx_ip1_j - vx_im1_j)/(2*dx))**2
        t3 = 2 * ((vx_i_jp1 - vx_i_jm1)/(2*dy)) * ((vy_ip1_j - vy_im1_j)/(2*dx))
        t4 = ((vy_i_jp1 - vy_i_jm1)/(2*dy))**2

        return p1 * (t1-t2-t3-t4)

    def next_step(self, dt):
        space = self.space
        grid = space.grid
        dx = space.dx
        dy = space.dy
        nx = space.nx
        ny = space.ny

        nu = self.nu
        rho = self.rho

        dtdx = (dt/dx)
        dtdy = (dt/dy)

        for i in range(nx):
            for j in range(ny):
                if i == 0:
                    self.next_space.grid[i][j].v.vx = 0
                    self.next_space.grid[i][j].v.vy = 1
                    self.next_space.grid[i][j].p = 10
                    continue

                if i == nx-1: 
                    self.next_space.grid[i][j].v.vx = self.next_space.grid[i-1][j].v.vx  
                    self.next_space.grid[i][j].v.vy = self.next_space.grid[i-1][j].v.vy
                    self.next_space.grid[i][j].p = -10  
                    continue

                if j == 0 or j == ny-1:  
                    self.next_space.grid[i][j].v.vx = 0 
                    self.next_space.grid[i][j].v.vy = 0
                    self.next_space.grid[i][j].p = self.next_space.grid[i][j-1].p if j == ny-1 else self.next_space.grid[i][j+1].p  
                    continue

                obstacle_radius = 5
                if (i-int(nx/2))**2 + (j-int(ny/2))**2 < obstacle_radius**2:  
                    self.next_space.grid[i][j].v.vx = 0  
                    self.next_space.grid[i][j].v.vy = 0
                    self.next_space.grid[i][j].p = 0  
                    continue
            

                curr_spacepoint = self.space.grid[i][j]
                vx = curr_spacepoint.v.vx
                vy = curr_spacepoint.v.vy
                p = curr_spacepoint.p

                vx_p = vx \
                    - vx * dtdx * (vx - grid[i-1][j].v.vx) \
                    - vy * dtdy * (vx - grid[i][j-1].v.vx) \
                    - (dtdx/(2*rho)) * (grid[i+1][j].p - grid[i-1][j].p) \
                    + nu * ((dtdx/dx)*(grid[i+1][j].v.vx + grid[i-1][j].v.vx - 2*vx)) \
                    + nu * ((dtdy/dy)*(grid[i][j+1].v.vx + grid[i][j-1].v.vx - 2*vx))

                vy_p = vy \
                    - vx * dtdx * (vy - grid[i-1][j].v.vy) \
                    - vy * dtdy * (vy - grid[i][j-1].v.vy) \
                    - (dtdy/(2*rho)) * (grid[i][j+1].p - grid[i][j-1].p) \
                    + nu * ((dtdx/dx)*(grid[i+1][j].v.vy + grid[i-1][j].v.vy - 2*vy)) \
                    + nu * ((dtdy/dy)*(grid[i][j+1].v.vy + grid[i][j-1].v.vx - 2*vy))
                
                p_p = (((grid[i+1][j].p + grid[i-1][j].p)*dy*dy + (grid[i][j+1].p + grid[i][j-1].p)*dx*dx)/(2 * (dx**2 +dy**2))) \
                    - self.calc_brackets(
                        dx, 
                        dy, 
                        dt,
                        rho,
                        grid[i+1][j].v.vx,
                        grid[i-1][j].v.vx,
                        grid[i][j+1].v.vx,
                        grid[i][j-1].v.vx,
                        grid[i][j+1].v.vy,
                        grid[i][j-1].v.vy,
                        grid[i+1][j].v.vy,
                        grid[i+1][j].v.vy,
                    )
                
                print("i,j: {}, {}".format(i, j))
                print("v  : {}, {}".format(vx, vy))
                print("v_p: {}, {}".format(vx_p, vy_p))
                print("-------------")
                
                self.next_space.grid[i][j].v.vx = vx_p
                self.next_space.grid[i][j].v.vy = vy_p
                self.next_space.grid[i][j].p = p

        return self.next_space.grid
    
    def solve(self, iters):
        for iter in range(iters):
            newgrid = self.next_step(self.dt)
            self.space.grid = newgrid

            if iter % 10 == 0:
                self.space.plot_pressure()


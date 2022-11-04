from scipy import sparse
from timesteppers import StateVector, CrankNicolson, RK22
import finite
import numpy as np

class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):
        self.react = self.Reaction(c, D, dx2)
        self.diffx = self.Diffusion_x(c, D, dx2)
        self.diffy = self.Diffusion_y(c, D, dy2)
        self.ts_react = RK22(self.react)
        self.ts_diffx = CrankNicolson(self.diffx, 0)
        self.ts_diffy = CrankNicolson(self.diffy, 1)
        self.dt = None
        self.iter = 0
        self.t = 0

    
    class Reaction:
        
        def __init__(self, c, D, dx2):
            self.X = StateVector([c], axis=0)
            N = c.shape[0]
            self.M = sparse.eye(N,N)
            self.L = 0*dx2.matrix
            self.F = lambda X: X.data*(1-X.data)
            
    class Diffusion_x:
        
        def __init__(self, c, D, dx2):
            self.X = StateVector([c], axis=0)
            N = c.shape[0]
            self.M = sparse.eye(N,N)
            self.L = -D*dx2.matrix
            self.F = lambda X: 0*X.data
            
            
    class Diffusion_y:
        
        def __init__(self, c, D, dy2):
            self.X = StateVector([c], axis=1)
            N = c.shape[0]
            self.M = sparse.eye(N,N)
            self.L = -D*dy2.matrix
            self.F = lambda X: 0*X.data
        

    def step(self, dt):
        # self.ts_react.step(dt/2)
        # self.ts_diffy.step(dt)
        # self.ts_diffx.step(dt)
        # self.ts_react.step(dt/2)
        self.ts_diffy.step(dt/2)
        self.ts_diffx.step(dt/2)
        self.ts_react.step(dt)
        self.ts_diffy.step(dt/2)
        self.ts_diffx.step(dt/2)

        self.dt = dt
        self.t += dt
        self.iter += 1


class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        grid_x = domain.grids[0]
        grid_y = domain.grids[1]
        dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, 0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, 1)
        dx2 = finite.DifferenceUniformGrid(2, spatial_order, grid_x, 0)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, grid_y, 1)
        
        self.react = self.Reaction(u, v, dx, dy)
        self.diffx = self.Diffusion_x(u, v, nu, dx2)
        self.diffy = self.Diffusion_y(u, v, nu, dy2)
        self.ts_react = RK22(self.react)
        self.ts_diffx = CrankNicolson(self.diffx, 0)
        self.ts_diffy = CrankNicolson(self.diffy, 1)
        self.dt = None
        self.iter = 0
        self.t = 0

    class Reaction:
        
        def __init__(self, u, v, dx, dy):
            self.X = StateVector([u, v])
            N = len(u)
            I = sparse.eye(N,N)
            Z = sparse.csr_matrix((N, N))
            
            M00 = I
            M01 = Z
            M10 = Z
            M11 = I
            self.M = sparse.bmat([[M00, M01],
                                  [M10, M11]])            
            
            L00 = (u @ dx.matrix) + (v @ dy.matrix)
            L01 = Z
            L10 = Z
            L11 = (u @ dx.matrix) + (v @ dy.matrix)
            self.L = sparse.bmat([[L00, L01],
                                  [L10, L11]])
            
            self.F = lambda X: 0*X.data
            
    class Diffusion_x:
        
        def __init__(self, u, v, nu, dx2):
            self.X = StateVector([u, v])
            N = len(u)
            I = sparse.eye(N,N)
            Z = sparse.csr_matrix((N, N))
            
            M00 = I
            M01 = Z
            M10 = Z
            M11 = I
            self.M = sparse.bmat([[M00, M01],
                                  [M10, M11]])

            L00 = -nu*dx2.matrix
            L01 = Z
            L10 = Z
            L11 = -nu*dx2.matrix
            self.L = sparse.bmat([[L00, L01],
                                  [L10, L11]])
            
            self.F = lambda X: 0*X.data
            
    class Diffusion_y:
        
        def __init__(self, u, v, nu, dy2):
            self.X = StateVector([u, v])
            N = len(u)
            I = sparse.eye(N,N)
            Z = sparse.csr_matrix((N, N))
            
            M00 = I
            M01 = Z
            M10 = Z
            M11 = I
            self.M = sparse.bmat([[M00, M01],
                                  [M10, M11]])
            
            L00 = -nu*dy2.matrix
            L01 = Z
            L10 = Z
            L11 = -nu*dy2.matrix
            self.L = sparse.bmat([[L00, L01],
                                  [L10, L11]])
            
            self.F = lambda X: 0*X.data
    
    def step(self, dt):
        # self.ts_react.step(dt/2)
        # self.ts_diffx.step(dt)
        # self.ts_diffy.step(dt)
        # self.ts_react.step(dt/2)
        self.ts_diffy.step(dt/2)
        self.ts_diffx.step(dt/2)
        self.ts_react.step(dt)
        self.ts_diffy.step(dt/2)
        self.ts_diffx.step(dt/2)

        self.dt = dt
        self.t += dt
        self.iter += 1


class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        
        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, p0):     
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N,N)
        Z = sparse.csr_matrix((N,N))
        
        if np.isscalar(rho0):
            M00 = rho0*I
        else:
            D = sparse.diags(rho0)
            M00 = D @ I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])
        
        L00 = Z
        L01 = d.matrix
        if np.isscalar(p0):
            L10 = p0*d.matrix
        else:
            D = sparse.diags(p0)
            L10 = D @ d
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        
        self.F = lambda X: 0*X.data

class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        N = len(c)
        
        self.M = sparse.eye(N,N)
        self.L = -D*d2.matrix
        self.F = lambda X: X.data*(c_target - X.data)

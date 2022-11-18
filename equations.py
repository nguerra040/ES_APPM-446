from scipy import sparse
from timesteppers import StateVector, CrankNicolson, RK22
import timesteppers
import finite
import numpy as np
import math
#from field import Field, FieldSystem
#from timesteppers import PredictorCorrector
#from spatial import FiniteDifferenceUniformGrid, Left, Right


class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = timesteppers.StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X.data*(1-X.data)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J


class BurgersFI:
    
    def __init__(self, u, nu, spatial_order, grid):
        self.X = timesteppers.StateVector([u])
        d = finite.DifferenceUniformGrid(1, spatial_order, grid)
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        N = len(u)
        I = sparse.eye(N)
        
        self.M = I
        self.L = -nu*d2.matrix
        self.F = lambda X: -np.multiply(X.data, (d @ X.data))
        self.J = lambda X: - sparse.diags(d @ X.data) - sparse.diags(X.data) @ d.matrix


class ReactionTwoSpeciesDiffusion:
    
    def __init__(self, X, D, r, spatial_order, grid):
        self.X = X # timesteppers.StateVector([c1, c2])
        self.r = r
        d = finite.DifferenceUniformGrid(1, spatial_order, grid)
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.X.scatter()
        c1, c2 = self.X.variables
        N = len(c1)
        I = sparse.eye(N)
        Z = sparse.csr_matrix((N, N))
        
        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = -D*d2.matrix
        L01 = Z
        L10 = Z
        L11 = -D*d2.matrix
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])
        
        def F(X):
            X.scatter()
            c1, c2 = X.variables
            F0 = np.multiply(c1, (1-c1-c2))
            F1 = np.multiply(self.r*c2, (c1-c2))
            return np.concatenate((F0, F1), axis=0)
        self.F = F
            
        def J(X):
            X.scatter()
            c1, c2 = X.variables
            J00 = sparse.diags(1 - 2 * c1 - c2)
            J01 = sparse.diags(-c1)
            J10 = sparse.diags(self.r*c2)
            J11 = sparse.diags(self.r*c1-2*self.r*c2)
            return sparse.bmat([[J00, J01],
                                [J10, J11]])
        self.J = J
        

    
class DiffusionBC:
    
    def __init__(self, c, D, spatial_order, domain):
        grid_x = domain.grids[0]
        grid_y = domain.grids[1]
        dx2 = finite.DifferenceUniformGrid(2, spatial_order, grid_x, 0)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, grid_y, 1)
        
        self.diffx = self.Diffusion_x(c, D, dx2, grid_x, spatial_order)
        self.diffy = self.Diffusion_y(c, D, dy2, grid_x, spatial_order)
        self.ts_diffx = CrankNicolson(self.diffx, 0)
        self.ts_diffy = CrankNicolson(self.diffy, 1)
        self.dt = None
        self.iter = 0
        self.t = 0
    
    class Diffusion_x:
        def __init__(self, c, D, dx2, grid_x, spatial_order):
            self.X = StateVector([c], axis=0)
            N = c.shape[0]
            
            M = sparse.eye(N,N)
            M = M.tocsr()
            # Boundary conditions (but no time dependent BC)
            M[0,:] = 0
            M[-1,:] = 0
            self.M = M
            
            L = -D*dx2.matrix
            L = L.tocsr()
            L[0,:] = 0
            L[-1,:] = 0
            # Left BC
            L[0, 0] = 1
            
            # Derivative w.r.t. x BC on right boundary
            num_terms = spatial_order+1
            b = np.zeros(num_terms)
            b[1]=1
            
            A = np.zeros((num_terms, num_terms))
            A[0,:] = 1
            for i in range(1, num_terms):
                for j in range(num_terms-1):
                    A[i,j] = ((-(num_terms-j-1)*grid_x.dx)**i)/math.factorial(i)
            coefs = np.linalg.solve(A, b)
            
            for i in range(len(coefs)):
                L[-1, -(num_terms-i)] = coefs[i]
            L.eliminate_zeros()
            self.L = L
            
            self.F = lambda X: 0*X.data
            
    class Diffusion_y:
        def __init__(self, c, D, dy2, grid_x, spatial_order):
            self.X = StateVector([c], axis=1)
            N = c.shape[0]
            self.M = sparse.eye(N,N)           
            self.L = -D*dy2.matrix
            self.F = lambda X: 0*X.data
            
    def step(self, dt):
        self.ts_diffy.step(dt)
        self.ts_diffx.step(dt)

        self.dt = dt
        self.t += dt
        self.iter += 1    
        
# class Wave2DBC:
    
#     def __init__(self, u, v, p, spatial_order, domain):
#         grid_x = domain.grids[0]
#         grid_y = domain.grids[1]
#         dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, 0)
#         dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, 1)
        
#         self.wavex = self.Wave_x(u, v, p, dx)
#         self.wavey = self.Wave_y(u, v, p, dy)
#         self.ts_wavex = RK22(self.wavex)
#         self.ts_wavey = RK22(self.wavey)
#         self.dt = None
#         self.iter = 0
#         self.t = 0

    
#     class Wave_x:
#         def __init__(self, u, v, p, dx):
#             self.X = StateVector([u, v, p])
            
#             def F(X):
#                 u_len_row = np.shape(X.variables[0][:][:])[0]
#                 v_len_row = np.shape(X.variables[1][:][:])[0]
#                 F0 = -dx.matrix @ X.data[(u_len_row+v_len_row):,:]
#                 F1 = 0*X.data[u_len_row:(u_len_row+v_len_row),:]
#                 F2 = -dx.matrix @ X.data[:u_len_row,:]
#                 return np.concatenate((F0, F1, F2), axis=0)
#             self.F = F
            
#             def BC(X):
#                 u_len_row = np.shape(X.variables[0][:][:])[0]
#                 v_len_row = np.shape(X.variables[1][:][:])[0]
#                 X.data[:(u_len_row+v_len_row),0] = 0
#                 X.data[:(u_len_row+v_len_row),-1] = 0
#             self.BC = BC
            
#     class Wave_y:
#         def __init__(self, u, v, p, dy):
#             self.X = StateVector([u, v, p])
#             def F(X):
#                 u_len_row = np.shape(X.variables[0][:][:])[0]
#                 v_len_row = np.shape(X.variables[1][:][:])[0]
#                 F0 = 0*X.data[:u_len_row,:]
#                 F1 = -dy.matrix @ X.data[(u_len_row+v_len_row):,:]
#                 F2 = -dy.matrix @ X.data[u_len_row:(u_len_row+v_len_row),:]
#                 return np.concatenate((F0, F1, F2), axis=0)
#             self.F = F
    
#     def step(self, dt):
#         self.ts_wavey.step(dt)
#         self.ts_wavex.step(dt)

#         self.dt = dt
#         self.t += dt
#         self.iter += 1

# class Wave2DBC:
    
#     def __init__(self, u, v, p, spatial_order, domain):
#         grid_x = domain.grids[0]
#         grid_y = domain.grids[1]
#         dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, 0)
#         dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, 1)
        
#         self.X = StateVector([u, v, p])
        
#         def F(X):
#             u_len_row = np.shape(X.variables[0][:][:])[0]
#             v_len_row = np.shape(X.variables[1][:][:])[0]
#             F0 = -dx.matrix @ X.data[(u_len_row+v_len_row):,:]
#             F1 = X.data[(u_len_row+v_len_row):,:] @ dy.matrix
#             F2 = -dx.matrix @ X.data[:u_len_row,:] + X.data[u_len_row:(u_len_row+v_len_row),:] @ dy.matrix
#             return np.concatenate((F0, F1, F2), axis=0)
#         self.F = F
        
#         def BC(X):
#             u_len_row = np.shape(X.variables[0][:][:])[0]
#             v_len_row = np.shape(X.variables[1][:][:])[0]
#             X.data[:(u_len_row+v_len_row),0] = 0
#             X.data[:(u_len_row+v_len_row),-1] = 0
#         self.BC = BC
        
class Wave2DBC:
    
    def __init__(self, u, v, p, spatial_order, domain):
        grid_x = domain.grids[0]
        grid_y = domain.grids[1]
        dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, 0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, 1)
        
        self.X = StateVector([u, v, p])
        
        def F(X):
            X.scatter()
            u, v, p = X.variables
            F0 = -dx.matrix @ p
            F1 = p @ dy.matrix
            F2 = -dx.matrix @ u + v @ dy.matrix
            return np.concatenate((F0, F1, F2), axis=0)
        self.F = F
        
        def BC(X):
            X.scatter()
            u, v, p = X.variables
            # u[:,0] = 0
            # u[:,-1] = 0
            u[0, :] = 0
            u[-1, :] = 0
            X.gather()
        self.BC = BC


class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):
        self.react = self.Reaction(c)
        self.diffx = self.Diffusion_x(c, D, dx2)
        self.diffy = self.Diffusion_y(c, D, dy2)
        self.ts_react = RK22(self.react)
        self.ts_diffx = CrankNicolson(self.diffx, 0)
        self.ts_diffy = CrankNicolson(self.diffy, 1)
        self.dt = None
        self.iter = 0
        self.t = 0

    
    class Reaction:
        
        def __init__(self, c):
            self.X = StateVector([c])
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
        self.ts_react.step(dt/2)
        self.ts_diffy.step(dt)
        self.ts_diffx.step(dt)
        self.ts_react.step(dt/2)

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
        
        self.advectx = self.Advection_x(u, v, dx)
        self.advecty = self.Advection_y(u, v, dy)
        self.diffx = self.Diffusion_x(u, v, nu, dx2)
        self.diffy = self.Diffusion_y(u, v, nu, dy2)
        self.ts_advectx = RK22(self.advectx)
        self.ts_advecty = RK22(self.advecty)
        self.ts_diffx = CrankNicolson(self.diffx, 0)
        self.ts_diffy = CrankNicolson(self.diffy, 1)
        self.dt = None
        self.iter = 0
        self.t = 0

    class Advection_x:
        
        def __init__(self, u, v, dx):
            self.X = StateVector([u, v])

            def F(X):
                len_u = np.shape(X.data)[0]//2
                F0 = -X.data[:len_u] * (dx @ X.data[:len_u])
                F1 = -X.data[:len_u] * (dx @ X.data[len_u:])
                return np.concatenate((F0, F1), axis=0)
            self.F = F
            
    class Advection_y:
        
        def __init__(self, u, v, dy):
            self.X = StateVector([u, v])

            def F(X):
                len_u = np.shape(X.data)[0]//2
                # F0 = X.data[len_u:] * (X.data[:len_u] @ dy.matrix)
                # F1 = X.data[len_u:] * (X.data[len_u:] @ dy.matrix)
                F0 = -X.data[len_u:] * (dy @ X.data[:len_u])
                F1 = -X.data[len_u:] * (dy @ X.data[len_u:])
                return np.concatenate((F0, F1), axis=0)
            self.F = F
            
    class Diffusion_x:
        
        def __init__(self, u, v, nu, dx2):
            self.X = StateVector([u, v], axis=0)
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
            self.X = StateVector([u, v], axis=1)
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
                
        
        self.ts_advectx.step(dt/2)
        self.ts_advecty.step(dt/2)
        self.ts_diffx.step(dt)
        self.ts_diffy.step(dt)
        self.ts_advecty.step(dt/2)
        self.ts_advectx.step(dt/2)

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

"""
ES_APPM 446 HW 3

Author: Nicolas Guerra
Due Date: 10/16/22
"""
import numpy as np
import scipy.sparse as sparse
import math
import scipy.sparse.linalg as spla
from scipy.linalg import lu_factor, lu_solve

class Timestepper:

    def __init__(self, u, f):
        self.t = 0
        self.iter = 0
        self.u = u
        self.func = f
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1
        
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ForwardEuler(Timestepper):

    def _step(self, dt):
        return self.u + dt*self.func(self.u)


class LaxFriedrichs(Timestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.func(self.u)


class Leapfrog(Timestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.func(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.func(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, func1, func2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = func1
        self.f2 = func2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(Timestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)         
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt):
        k = np.zeros((len(self.u), self.stages))
        k[:,0] = self.func(self.u)
        for i in range(1, self.stages): 
            argument = np.copy(self.u)
            for j in range(i): # only worry about below diagonal since assignment is for explicit schemes
                argument += self.a[i,j]*dt*k[:,j]
            k[:,i] = self.func(argument)
        return self.u + dt*(k @ self.b)


class AdamsBashforth(Timestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        
        # C*a=b where 'a' are the coef. with a0 at the bottom
        C = np.zeros((steps,steps))
        b = np.zeros((steps,1))
        for i in range(steps):
            for j in range(steps):
                C[i,j] = (-(steps-1-j))**i
            b[i] = 1/(i+1)
        a = np.linalg.solve(C, b)
        self.a = np.copy(a)
        
        # Get first s-1 steps with multistage scheme
        # The first column of u_archives is the newest column, self.u
        u_archives = np.zeros((len(self.u),steps))
        u_archives[:,0] = np.copy(self.u)
        stages = 3
        a = np.array([[ 0, 0, 0],
        [1/2, 0, 0],
        [ -1, 2, 0]])
        b = np.array([1, 4, 1])/6
        for i in range(1,steps):
            ts = Multistage(u_archives[:,i-1], self.func, stages, a, b)
            ts.step(self.dt/10)
            u_archives[:,i] = np.copy(ts.u)
        self.u_archives = np.copy(u_archives)
        self.u = np.copy(ts.u)

    def _step(self, dt):
        combine = np.zeros((np.shape(self.u)))
        for i in range(self.steps):
            combine += self.a[-(i+1)]*self.func(self.u_archives[:,i])
        new_u = self.u + dt*combine
        self.u_archives[:,1:] = np.copy(self.u_archives[:,:-1])
        self.u_archives[:,0] = np.copy(new_u)
        return new_u
        
# IMPLICIT METHODS
class BackwardEuler(Timestepper):
    def __init__(self, u, L):
        super().__init__(u, L)
        N = len(u)
        self.I = sparse.eye(N, N)
    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.u)
    
class CrankNicolson(Timestepper):
    def __init__(self, u, L_op):
        super().__init__(u, L_op)
        N = len(u)
        self.I = sparse.eye(N, N)
    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.func.matrix
            self.RHS = self.I + dt/2*self.func.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        return self.LU.solve(self.RHS @ self.u)
    
class BackwardDifferentiationFormula(Timestepper):
    def __init__(self, u, L_op, steps):
        super().__init__(u, L_op)
        self.u = u
        self.L_op = L_op
        self.steps = steps
        self.dt = None
        self.dt_array = np.array([])
        self.current_total_steps = 1
        self.u_archives = np.zeros((len(self.u),self.steps))
        # First column of u_archives is most recent
        self.u_archives[:,0] = np.copy(self.u)
        
    def _step(self, dt):
        if dt != self.dt:
            self.dt = np.copy(dt)
        if self.current_total_steps < self.steps:
            # Append most recent to top
            self.dt_array = np.append(self.dt, self.dt_array)
            # Let's compute coefficient vector of ai and B0 where x=[a1,...,as,B0]:
            A = np.zeros((self.current_total_steps+1,self.current_total_steps+1))
            A[0,:] = 1
            for i in range(1, self.current_total_steps+1):
                for j in range(1, self.current_total_steps+1):
                    A[i,j] = (sum(self.dt_array[:j])**i)/math.factorial(i)
            b = np.zeros(self.current_total_steps+1)
            b[1] = 1
            lu, piv = lu_factor(A)
            # x is the coefficients with a0 at the top
            x = lu_solve((lu, piv), b)
            self.x = -1*np.copy(x)
            
            RHS = -1*(self.u_archives[:,:self.current_total_steps] @ self.x[1:])
            LHS =  self.x[0]*np.identity(len(self.u)) - self.L_op.matrix
            lu, piv = lu_factor(LHS)
            new_u = lu_solve((lu, piv), RHS)
            self.u_archives[:, 1:] = np.copy(self.u_archives[:, :-1])
            self.u_archives[:, 0] = np.copy(new_u)
            self.current_total_steps += 1
            return new_u
            
        else:
            # Append most recent to top
            self.dt_array = np.append(self.dt, self.dt_array)
            # Let's compute coefficient vector of ai and B0 where x=[a1,...,as,B0]:
            A = np.zeros((self.steps+1,self.steps+1))
            A[0,:] = 1
            for i in range(1, self.steps+1):
                for j in range(1, self.steps+1):
                    A[i,j] = (sum(self.dt_array[:j])**i)/math.factorial(i)
            b = np.zeros(self.steps+1)
            b[1] = 1
            lu, piv = lu_factor(A)
            # x is the coefficients with a0 at the top
            x = lu_solve((lu, piv), b)
            self.x = -1*np.copy(x)
            
            RHS = -1*(self.u_archives @ self.x[1:])
            LHS =  self.x[0]*np.identity(len(self.u)) - self.L_op.matrix
            lu, piv = lu_factor(LHS)
            new_u = lu_solve((lu, piv), RHS)
            self.u_archives[:, 1:] = np.copy(self.u_archives[:, :-1])
            self.u_archives[:, 0] = np.copy(new_u)
            return new_u

################################## IMEX ####################################
class StateVector:
    
    def __init__(self, variables):
        var0 = variables[0]
        self.N = len(var0)
        size = self.N*len(variables)
        self.data = np.zeros(size)
        self.variables = variables
        self.gather()
    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[i*self.N:(i+1)*self.N], var)
    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[i*self.N:(i+1)*self.N])
            
class IMEXTimestepper:
    def __init__(self, eq_set):
        self.t = 0
        self.iter = 0
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.dt = None
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)
    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.t += dt
        self.iter += 1
        
class Euler(IMEXTimestepper):
    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        self.dt = dt
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)
    
class CNAB(IMEXTimestepper):
    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt
            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)
        
class BDFExtrapolate(IMEXTimestepper):
    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.current_total_steps = 1
        self.dt_array = np.array([])
        # oldest X is at the top so X_n-1 (current) is at bottom
        self.X_archives = [self.X]
        self.FX_archives = []
        
    
    def _step(self, dt):
        self.X_archives.append(self.X.data)
        self.FX_archives.append(self.F(self.X))
        if dt != self.dt:
            self.dt = np.copy(dt)
        if self.current_total_steps < self.steps:
            # solve first iterations with lower order method
            # Append most recent dt to top
            self.dt_array = np.append(self.dt, self.dt_array)
            
            # compute ai coefficients
            A = np.zeros((self.current_total_steps+1, self.current_total_steps+1))
            A[0,:] = 1
            for i in range(1, self.current_total_steps+1):
                for j in range(1, self.current_total_steps+1):
                    A[i,j] = ((-j*dt)**i)/math.factorial(i)
            b = np.zeros(self.current_total_steps+1)
            b[1] = 1
            lu, piv = lu_factor(A)
            # a is the coefficients with a0 at the top
            a = lu_solve((lu, piv), b)
            
            # compute bi coefficients
            A = np.zeros((self.current_total_steps, self.current_total_steps))
            A[0,:] = 1
            for i in range(1, self.current_total_steps):
                for j in range(self.current_total_steps):
                    A[i,j] = ((-(j+1)*dt)**i)/math.factorial(i)
            b = np.zeros(self.current_total_steps)
            b[0] = 1
            lu, piv = lu_factor(A)
            # bb is the coefficients with b1 at the top
            bb = lu_solve((lu, piv), b)
            
            LHS = self.M*a[0] + self.L
            
            First_RHS_Term = bb[0]*self.FX_archives[-1]
            for i in range(1, self.current_total_steps):
                First_RHS_Term += bb[i]*self.FX_archives[-(i+1)]
                
            Second_RHS_Term = a[1]*self.X_archives[-1]
            for i in range(2, self.current_total_steps+1):
                Second_RHS_Term += a[i]*self.X_archives[-i]
                
            RHS = First_RHS_Term - (self.M @ Second_RHS_Term)
            
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            new_X = self.LU.solve(RHS)
            self.current_total_steps += 1
            return new_X
        
        else:
            # Append most recent dt to top
            self.dt_array = np.append(self.dt, self.dt_array)
            
            # compute ai coefficients
            A = np.zeros((self.steps+1, self.steps+1))
            A[0,:] = 1
            for i in range(1, self.steps+1):
                for j in range(1, self.steps+1):
                    A[i,j] = ((-j*dt)**i)/math.factorial(i)
            b = np.zeros(self.steps+1)
            b[1] = 1
            lu, piv = lu_factor(A)
            # a is the coefficients with a0 at the top
            a = lu_solve((lu, piv), b)
            
            # compute bi coefficients
            A = np.zeros((self.steps, self.steps))
            A[0,:] = 1
            for i in range(1, self.steps):
                for j in range(self.steps):
                    A[i,j] = ((-(j+1)*dt)**i)/math.factorial(i)
            b = np.zeros(self.steps)
            b[0] = 1
            lu, piv = lu_factor(A)
            
            # bb is the coefficients with b1 at the top
            bb = lu_solve((lu, piv), b)
            
            LHS = self.M*a[0] + self.L
            
            First_RHS_Term = bb[0]*self.FX_archives[-1]
            for i in range(1, self.steps):
                First_RHS_Term += bb[i]*self.FX_archives[-(i+1)]
                
            Second_RHS_Term = a[1]*self.X_archives[-1]
            for i in range(2, self.steps+1):
                Second_RHS_Term += a[i]*self.X_archives[-i]
                
            RHS = First_RHS_Term - (self.M @ Second_RHS_Term)
            
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
            new_X = self.LU.solve(RHS)
            return new_X
            
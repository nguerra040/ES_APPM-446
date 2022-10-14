# -*- coding: utf-8 -*-
"""
ES_APPM 446 HW2
Due Date: 10/7/22

@author: Nicolas Guerra
"""
import numpy as np
from scipy.special import factorial
from scipy import sparse

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length # h = length/N
        self.N = N # Number of elements


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class DifferenceUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type

        if self.stencil_type == 'centered': # For centered, the sum of both orders must be odd
            if (self.derivative_order+self.convergence_order)%2 == 0:
                # change convergence order because if not, central diff. won't work
                self.convergence_order = self.convergence_order + 1
            b = np.zeros(self.derivative_order+self.convergence_order)
            b[self.derivative_order] = 1
            h = grid.values[1] # assuming first point is 0
            S = np.zeros((len(b),len(b)))

            # Fill in S
            for k in range(S.shape[0]): # row
                center = (S.shape[1]-1)//2 # make center j be 0
                offsets = np.arange(S.shape[1]) - center
                j_not_centered = 0 # python thinks index -1 is last column
                for j in offsets: # column
                    S[k,j_not_centered] = ((j*h)**k)/factorial(k)
                    j_not_centered += 1
            
            # Get stencil
            a = np.linalg.inv(S) @ b

            # Create D matrix
            D = sparse.diags(a, offsets=offsets, shape = [grid.N, grid.N])
            D = D.tocsr()
            n_cols_to_change = (len(a)-1)//2
            for i in range(n_cols_to_change):
                D[i, -(n_cols_to_change-i):] = a[:n_cols_to_change-i]
                D[-(1+i), :n_cols_to_change-i] = a[n_cols_to_change+1+i:]
            self.matrix = D    
            
    def __matmul__(self, other):
        return self.matrix @ other


class DifferenceNonUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        
        if stencil_type == 'centered':
            if (self.derivative_order+self.convergence_order)%2 == 0:
                # change convergence order because if not, central diff. won't work
                self.convergence_order = self.convergence_order + 1
            b = np.zeros(self.derivative_order+self.convergence_order)
            b[self.derivative_order] = 1
            D = np.zeros((len(grid.values), len(grid.values)))
            
            full_row_start = (len(b)-1)//2
            col = 0 # column index when inputing stencil
            for i in range(full_row_start, len(grid.values)-full_row_start):
                # Find a different stencil for each full row (no flopping to other side)
                S = np.zeros((len(b),len(b)))
                # Fill in S
                for k in range(S.shape[0]): # row
                    center = (S.shape[1]-1)//2 # make center j be 0
                    offsets = np.arange(S.shape[1]) - center
                    j_not_centered = 0 # python thinks index -1 is last column so this is just for indexing
                    for j in offsets: # column
                        # h = abs(grid.values[i+j]-grid.values[i])
                        # S[k,j_not_centered] = ((j*h)**k)/factorial(k)
                        h = grid.values[i+j]-grid.values[i]
                        S[k,j_not_centered] = ((h)**k)/factorial(k)
                        j_not_centered += 1
                        
                # Get stencil
                a = np.linalg.inv(S) @ b
                
                # Input stencil in D
                D[i, col:col+len(b)] = a
                col += 1

            # Find stencil for remaining upper rows
            for i in range((len(b)-1)//2):
                S = np.zeros((len(b),len(b)))
                # Fill in S
                for k in range(S.shape[0]): # row
                    center = (S.shape[1]-1)//2 # make center j be 0
                    offsets = np.arange(S.shape[1]) - center
                    j_not_centered = 0 # python thinks index -1 is last column so this is just for indexing
                    for j in offsets: # column
                        if i+j < 0:
                            h = -(grid.length-grid.values[i+j])-grid.values[i]
                            #h = grid.values[i+j]-grid.values[i]
                        else:
                            h = grid.values[i+j]-grid.values[i]
                        S[k,j_not_centered] = ((h)**k)/factorial(k)
                        j_not_centered += 1
                        
                # Get stencil
                a = np.linalg.inv(S) @ b
                
                # Input stencil in D
                wing = (len(b)-1)//2 + 1 #size of stencil wing including center
                D[i, :wing+i] = a[wing-i-1:]
                D[i, -(wing-i-1):] = a[:wing-1-i]
                
            shift = 0 # used to indicate where to insert stencil
            # Find stencil for remaining lower rows
            for i in range(len(grid.values)-(len(b)-1)//2,len(grid.values)): # last rows
                S = np.zeros((len(b),len(b)))
                # Fill in S
                for k in range(S.shape[0]): # row
                    center = (S.shape[1]-1)//2 # make center j be 0
                    offsets = np.arange(S.shape[1]) - center
                    j_not_centered = 0 # python thinks index -1 is last column so this is just for indexing
                    for j in offsets: # column
                        if i+j > len(grid.values)-1: # i+j can't be greater than it's max index
                            h = grid.length+grid.values[i+j-len(grid.values)]-grid.values[i]
                        else:
                            h = grid.values[i+j]-grid.values[i]
                        S[k,j_not_centered] = ((h)**k)/factorial(k)
                        j_not_centered += 1
                        
                # Get stencil
                a = np.linalg.inv(S) @ b
                # Input stencil in D
                D[i, :1+shift] = a[len(b)-1-shift:]
                D[i, -(len(b)-1-shift):] = a[:len(b)-1-shift]
                shift += 1
                
            self.matrix = D
                

    def __matmul__(self, other):
        return self.matrix @ other


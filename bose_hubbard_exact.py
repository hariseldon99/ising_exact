#!/usr/bin/env python
"""
Created on May 14 2015

@author: Analabha Roy (daneel@utexas.edu)

"""
import numpy as np
from pprint import pprint
from itertools import combinations, product
from scipy.integrate import odeint
from math import factorial
from mpi4py import MPI
from scipy.sparse import csr_matrix, lil_matrix

"""
Default Parameters are entered here
"""
#Number of lattice sites
lattice_size = 3   
#Number of particles
particle_no = 3

verbose = False

#Drive parameters
amp = 1.0
freq = 0.0
t_init = 0.0 # Initial time
t_final = 1.0 # Final time
n_steps = 100 # Number of time steps

desc = """Dynamics by exact diagonalization of 
		1d nearest neighbour bose hubbard model with periodic drive""" 

 
#def jac(y, t0, params):
#  return jacmat

#def func(y, t0, params):
#  return np.dot(params.jac, y)

#def evolve_numint(hamilt,times,initstate):
#  (rows,cols) = hamilt.hamiltmat.shape
#  fulljac = np.zeros((2*rows,2*cols), dtype="float64")
#  fulljac[0:rows, 0:cols] = hamilt.hamiltmat.imag
#  fulljac[0:rows, cols:] = hamilt.hamiltmat.real
#  fulljac[rows:, 0:cols] = -hamilt.hamiltmat.real
#  fulljac[rows:, cols:] = hamilt.hamiltmat.imag
  
#  psi_t = odeint(func, np.concatenate((initstate, np.zeros(rows))),\
#    times, args=(fulljac,), Dfun=jac)
#  return psi_t[:,0:rows] + (1.j) * psi_t[:, rows:]

def run_dyn():
    comm = MPI.COMM_WORLD
    if verbose:
        print "Executing diagonalization with parameters:"
        pprint(vars(params), depth=1)
    else:
        print "Starting run ..."
    #Dimensionality of the hilbert space
    m,n = lattice_size, particle_no    
    d = factorial(n+m-1)/(factorial(n) * factorial(m-1))
    
    #Build the dxm-size fock state matrix as per ref. Do this SEQUENTIALLY
    all_fockstates =  lil_matrix((d, m), dtype=np.float64)
    #Set thefirst row to the highest fock state as per ref
    all_fockstates[0,0] = n
    for i in xrange(d):
        if i != 0:
            prev_fockstate = all_fockstates[i-1,:].toarray().flatten()
            k = prev_fockstate.nonzero()[0][-1] 
            all_fockstates[i,:k] = prev_fockstate[:k]
            all_fockstates[i,k] = prev_fockstate[k] - 1.0
            if k < m-1:
                all_fockstates[i,k+1] = n - np.sum(all_fockstates[i,:k].toarray())
    
    all_fockstates = all_fockstates.tocsr()
    pprint(all_fockstates.toarray())        
    #Build kinetic energy matrix, i.e. \Sum_i (c_i^{\dagger}c_{i+1} + h.c.) 
    #kemat = 
    
 
    #initstate =  
    #dt = (t_final-t_init)/(n_steps-1.0)
    #t_output = np.arange(t_init, t_final, dt)
  
    #psi_t = evolve_numint(h, t_output, initstate)
  
    #print "\nDumping outputs to files ..."
    #print 'Done'

if __name__ == '__main__':
  run_dyn()
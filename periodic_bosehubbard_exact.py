#!/usr/bin/env python
"""
Created on May 14 2015

@author: Analabha Roy (daneel@utexas.edu)

Usage : ./periodic_bosehubbard_exact -h
"""
import argparse
import numpy as np
from pprint import pprint
from itertools import combinations, product
from scipy.integrate import odeint
from petsc4py import PETSc
from slepc4py import SLEPc

"""
Default Parameters are entered here
"""
#Number of lattice sites
M = 6   
#Number of particles
N = 3

t_init = 0.0 # Initial time
t_final = 1.0 # Final time
n_steps = 100 # Number of time steps

desc = """Dynamics by exact diagonalization of 
		1d nearest neighbour bose hubbard model with periodic drive""" 

def input():
  parser = argparse.ArgumentParser(description=desc)
  
  parser.add_argument('-m', '--lattice_size', type=np.int64,\
    help="size of the lattice", default=M)

  parser.add_argument('-n', '--particle_no', type=np.int64,\
    help="number of particles", default=N)
    
  
  parser.add_argument('-or', '--output_r',\
    help="response (density) output file", default="res_outfile.txt")
  parser.add_argument('-ov', '--output_var',\
    help="Number variance output file", default="var_outfile.txt")
    
  parser.add_argument("-v", '--verbose', action='store_true', \
    help="increase output verbosity")

  parser.add_argument('-pbc', '--periodic', \
    help="switch to periodic boundary conditions, default is open",\
      action="store_true")

  parser.add_argument('-a', '--amp', type=np.float64, \
    help="drive amplitude", default=1.0)
  parser.add_argument('-w', '--freq', type=np.float64, \
    help="drive frequency", default=0.0)
  
  return parser.parse_args()
 
def jac(y, t0, jacmat):
  return jacmat

def func(y, t0, jacmat):
  return np.dot(jac(y, t0, jacmat), y)

def hamiltonian(params):
    # grid size and spacing
    m, n  = 32, 32
    hx = 1.0/(m-1)
    hy = 1.0/(n-1)
    
    # create sparse matrix
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([m*n, m*n])
    A.setType('aij') # sparse
    A.setPreallocationNNZ(5)
    
    # precompute values for setting
    # diagonal and non-diagonal entries
    diagv = 2.0/hx**2 + 2.0/hy**2
    offdx = -1.0/hx**2
    offdy = -1.0/hy**2
    
    # loop over owned block of rows on this
    # processor and insert entry values
    Istart, Iend = A.getOwnershipRange()
    for I in range(Istart, Iend) :
        A[I,I] = diagv
        i = I//n    # map row number to
        j = I - i*n # grid coordinates
        if i> 0  : J = I-n; A[I,J] = offdx
        if i< m-1: J = I+n; A[I,J] = offdx
        if j> 0  : J = I-1; A[I,J] = offdy
        if j< n-1: J = I+1; A[I,J] = offdy
    
    # communicate off-processor values
    # and setup internal data structures
    # for performing parallel operations
    A.assemblyBegin()
    A.assemblyEnd()
    return A


def evolve_numint(hamilt,times,initstate):
  (rows,cols) = hamilt.hamiltmat.shape
  fulljac = np.zeros((2*rows,2*cols), dtype="float64")
  fulljac[0:rows, 0:cols] = hamilt.hamiltmat.imag
  fulljac[0:rows, cols:] = hamilt.hamiltmat.real
  fulljac[rows:, 0:cols] = -hamilt.hamiltmat.real
  fulljac[rows:, cols:] = hamilt.hamiltmat.imag
  
  psi_t = odeint(func, np.concatenate((initstate, np.zeros(rows))),\
    times, args=(fulljac,), Dfun=jac)
  return psi_t[:,0:rows] + (1.j) * psi_t[:, rows:]

def run_dyn(params):
  if params.verbose:
    print "Executing diagonalization with parameters:"
    pprint(vars(params), depth=1)
  else:
    print "Starting run ..."
 
  h = hamiltonian(params)  
  lsize = h.lattice_size
  lsq = lsize * lsize
  
  initstate =  
  dt = (t_final-t_init)/(n_steps-1.0)
  t_output = np.arange(t_init, t_final, dt)
  

  psi_t = evolve_numint(h, t_output, initstate)
  
  print "\nDumping outputs to files ..."
  data.dump_data()
  print 'Done'

if __name__ == '__main__':
  args_in = input()
  run_dyn(args_in)
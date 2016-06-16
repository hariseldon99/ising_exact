#!/usr/bin/env python
"""
Created on June 13 2016

@author: Analabha Roy (daneel@utexas.edu)

"""
from __future__ import division, print_function
import numpy as np
from pprint import pprint
from itertools import chain
from operator import sub
from scipy.integrate import odeint
from math import factorial
from petsc4py import PETSc
from scipy.sparse import lil_matrix, dia_matrix
from bisect import bisect_left

timesteps = 1000
verbose = False

#Verbosity function
def verboseprint(verbosity, *args):
    if verbosity:
        for arg in args:
            pprint(arg)
        print(" ")

def index(a, x):
    """
    Locate the leftmost value exactly equal to x
    """
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def boxings(n, k):
    """
    boxings(n, k) -> iterator
    Generate all ways to place n indistiguishable items into k
    distinguishable boxes
    """
    seq = [n] * (k-1)
    while True:
        yield map(sub, chain(seq, [n]), chain([0], seq))
        for i, x in enumerate(seq):
            if x:
                seq[:i+1] = [x-1] * (i+1)
                break
        else:
            return

def jac(y, t, params):
    """
    Synthesizes the Hamiltonian and returns it as a Jacobian
    of the Schroedinger Equation
    """
    return (-1j) * (1.0 + params.amp * np.cos(params.freq*t)) *\
                                                     params.h_ke + params.h_int
def func(y, t, params):
    """
    This is basically just Schroedinger's Equation
    """
    return jac(y, t, params).dot(y)

class ParamData:
    """Class that stores Hamiltonian matrices and 
       and system parameters. This class has no 
       methods other than the constructor.
    """
    def __init__(self, lattice_size=3, particle_no=3, amp=1.0, freq=1.0, \
                                 int_strength=1.0, disorder_strength=1.0,\
                                      mpicomm=PETSc.COMM_WORLD, verbose=False):
      """
       Usage:
       p = ParamData(lattice_size=3, particle_no=3, amp=1.0, freq=0.0, \
            int_strength=1.0, disorder_strength=1.0, mpicomm=comm)
		      
       All parameters (arguments) are optional.
       
       Parameters:
       lattice_size   	=  The size of your lattice as an integer.
       particle_no     =  The number of particles (bosons) in the lattice 
       amp             =  The periodic (cosine) drive amplitude. Defaults to 1.
       freq   	 =  The periodic (cosine) drive frequency. Defaults to 1.
       int_strength  =  The interaction strength U of the Bose Hubbard model.
                        Defaults to 1.
       disorder_strength =  Amplitude of the site disorder. Defaults to 1.
       mpicomm           =  MPI Communicator. Defaults to PETSc.COMM_WORLD
       verbose           =  Boolean for verbose output. Defaults to False

       Return value: 
       An object that stores all the parameters above. 
      """
      self.lattice_size = lattice_size
      self.particle_no = particle_no
      self.amp, self.freq = amp, freq
      self.int_strength = int_strength
      self.disorder_strength = disorder_strength
      self.comm = mpicomm
      self.verbose = verbose
      rank = self.comm.Get_rank()
      n,m = self.particle_no, self.lattice_size    
      #Field disorder
      h = np.random.uniform(-1.0, 1.0, m)
      #Interaction
      U = self.int_strength
      #Disorder strength
      alpha = self.disorder_strength
      tagweights = 100 * np.arange(m) + 3
      #Dimensionality of the hilbert space
      self.dimension = factorial(n+m-1)/(factorial(n) * factorial(m-1))
      d = self.dimension
      if rank == 0:
          #Build the dxm-size fock state matrix as per ref. 
          all_fockstates =  lil_matrix((d, m), dtype=np.float64)
          fockstate_tags = np.zeros(d)
          h_int_diag = np.zeros_like(fockstate_tags)
          for row_i, row in enumerate(boxings(n,m)):
              all_fockstates[row_i,:] = row
              row = np.array(row)
              fockstate_tags[row_i] = tagweights.dot(row)
              h_int_diag[row_i] = np.sum(row * (U * (row-1) - alpha * h))
          all_fockstates = all_fockstates.tocsr()
          #Build the interaction matrix i.e U\sum_i n_i (n_i-1) -\alpha h_i n_i
          self.h_int = dia_matrix((d, d), dtype=np.float64)
          self.h_int.setdiag(h_int_diag)
          #Sort the tags and store the original indices
          tag_inds = np.argsort(fockstate_tags)
          fockstate_tags = fockstate_tags[tag_inds]
          #Build kinetic energy matrix, i.e. \sum_i (c_i^{\dagger}c_{i+1} + h.c.) 
          self.h_ke =  lil_matrix((d, d), dtype=np.float64)
          for v, fock_v in enumerate(boxings(n,m)):
              fock_v = np.array(fock_v)
              tag = tagweights.dot(fock_v)
              #Search for this tag in the sorted array of tags
              w = index(fockstate_tags, tag)
              #Get the corresponding row index
              u = tag_inds[w]
              #This is -\sum_i \sqrt{(A_{vi}+1)A_{vi+1}} where A is the fockstate
              #The "roll" implements periodic boundary conditions
              elem = - np.sum(np.sqrt((fock_v + 1) * np.roll(fock_v,1)))
              self.h_ke[u,v] = elem
          self.h_ke = (self.h_ke+self.h_ke.T).tocsr()
      else:
          all_fockstates = None
          self.h_int = None
          self.h_ke = None
      self.h_int = self.comm.tompi4py().bcast(self.h_int, root=0)    
      self.h_ke  = self.comm.tompi4py().bcast(self.h_ke, root=0)

class FloquetMatrix:
    """Class that evaluates the Floquet Matrix of a time-periodically
       driven Bose Hubbard model
    """  
    def __init__(self, params):
        """
        This creates a distributed parallel dense unit matrix
        """
        d = params.dimension
        #Setup the Floquet Matrix in parallel
        self.fmat = PETSc.Mat()
        self.fmat.create(params.comm)
        locSize = PETSc.PETSC_DECIDE
        self.fmat.setSizes(((locSize, d), (locSize, d)), bsize=1)
        self.fmat.setType('dense')
        #Initialize it to unity
        diag = self.fmat.getDiagonal()
        diag.set(1.0)
        self.fmat.setDiagonal(diag)
        self.fmat.assemblyBegin()
        self.fmat.assemblyEnd()
    
    def evolve(self, params):
        """
        This evolves each column of the Floquet Matrix in time via the 
        periodically driven Bose Hubbard model. The Floquet Matrix 
        is updated after one time period
        """
        d = params.dimension
        times = np.linspace(0.0, 2.0 * np.pi/params.freq, num=timesteps)
        Istart, Iend = self.fmat.getOwnershipRange()
        for I in xrange(Istart, Iend):
            #Get the Ith row and evolve it
            psi_t = odeint(func, self.fmat.getRow(I),\
                                             times, args=(params,), Dfun=None)
            #Set the Ith row to the final state after evolution                                 
            self.fmat.setValuesLocal(I,np.arange(d),psi_t[-1])                                 
        self.fmat.assemblyBegin()
        self.fmat.assemblyEnd()

    def diagonalize(self):
        """
        This diagonalizes the Floquet Matrix after evolution.
        NOTE: COMPLETE THIS
        """
        pass
    
if __name__ == '__main__':
  p = ParamData(lattice_size=3, particle_no=3, amp=1.0, freq=0.0, \
            int_strength=1.0, disorder_strength=1.0)
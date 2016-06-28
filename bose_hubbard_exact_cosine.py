#!/usr/bin/env python
"""
Created on June 13 2016

@author: Analabha Roy (daneel@utexas.edu)

"""
from __future__ import division, print_function
import numpy as np
from pprint import pprint
from math import factorial
from itertools import chain
from operator import sub
from bisect import bisect_left
from scipy.sparse import lil_matrix, dia_matrix
from scipy.integrate import odeint
from petsc4py import PETSc
from slepc4py import SLEPc

timesteps = 100
petsc_int = np.int32 #petsc, by default. Uses 32-bit integers for indices
ode_dtype = np.float64 #scipy.odeint does not do complex numbers, so use this.

#Verbosity function
def verboseprint(verbosity, *args):
    if verbosity:
        for arg in args:
            pprint(arg, depth=2)
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

def drive(t, a, f):
    """
    Time dependent drive on hopping amplitude J
    """
    return 1.0 + a * np.cos(f*t)

def jac(y, t, p):
    """
    Synthesizes the Jacobian of the Schroedinger Equation
    """
    return drive(t, p.amp, p.freq) * p.jac_ke + p.jac_int
    
def func(y, t, p):
    """
    This is basically just Schroedinger's Equation
    """
    return jac(y, t, p).dot(y)

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
      n,m = self.particle_no, self.lattice_size    
      #Field disorder
      h = np.random.uniform(-1.0, 1.0, m)
      #Interaction
      U = self.int_strength
      #Disorder strength
      alpha = self.disorder_strength
      #Dimensionality of the hilbert space
      self.dimension = np.int(factorial(n+m-1)/(factorial(n) * factorial(m-1)))
      size = self.comm.Get_size()      
      assert self.dimension >= size, "There are fewer rows than MPI procs!"
      d = self.dimension
      rank = self.comm.Get_rank()
      tagweights = np.sqrt(100 * np.arange(m) + 3)
      if rank == 0:
          #Build the dxm-size fock state matrix as per ref. 
          all_fockstates =  lil_matrix((d, m), dtype=ode_dtype)
          fockstate_tags = np.zeros(d)
          h_int_diag = np.zeros_like(fockstate_tags)
          for row_i, row in enumerate(boxings(n,m)):
              all_fockstates[row_i,:] = row
              row = np.array(row)
              fockstate_tags[row_i] = tagweights.dot(row)
              h_int_diag[row_i] = np.sum(row * (U * (row-1) - alpha * h))
          #Build the interaction matrix i.e U\sum_i n_i (n_i-1) -\alpha h_i n_i
          data = np.array([np.concatenate((np.zeros(d), h_int_diag)),\
                                   np.concatenate((-h_int_diag, np.zeros(d)))])
          offsets = np.array([d,-d])                          
          self.jac_int = dia_matrix((data,offsets), shape=(2*d, 2*d),\
                                                              dtype=ode_dtype)
          #Sort the tags and store the original indices
          tag_inds = np.argsort(fockstate_tags)
          fockstate_tags = fockstate_tags[tag_inds]
          #Build hop of kinetic energy matrix, i.e. \sum_i c_i^{\dagger}c_{i+1}  
          h_ke =  lil_matrix((d, d), dtype=ode_dtype)
          for site in xrange(m):  
              next_site = 0 if site==m-1 else site+1 #Periodic bc
              for v, fock_v in enumerate(all_fockstates):
                  fock_v = fock_v.toarray().flatten()
                  #Note that, if any site has no particles, then one annihilation
                  #operator nullifies the whole state
                  if fock_v[next_site] != 0:
                      #Hop a particle in fock_v from i to i+1 & store.
                      fockv_hopped = np.copy(fock_v)
                      fockv_hopped[site] += 1
                      fockv_hopped[next_site] -= 1
                      #Tag this hopped vector
                      tag = tagweights.dot(fockv_hopped)
                      #Binary search for this tag in the sorted array of tags
                      w = index(fockstate_tags, tag) 
                      #Get the corresponding row index 
                      u = tag_inds[w]
                      #Add the hopping amplitude to the matrix element
                      h_ke[u,v] -= np.sqrt((fock_v[site]+1) * fock_v[next_site])          
          h_ke = (h_ke + h_ke.T) # Add the hermitian conjugate
          self.jac_ke = lil_matrix((2*d, 2*d), dtype=ode_dtype)
          self.jac_ke[:d,d:] = h_ke
          self.jac_ke[d:,:d] = -h_ke
          self.jac_ke = self.jac_ke.tocsr()
      else:
          all_fockstates = None
          self.jac_int = None
          self.jac_ke = None
      self.jac_int = self.comm.tompi4py().bcast(self.jac_int, root=0)    
      self.jac_ke  = self.comm.tompi4py().bcast(self.jac_ke, root=0)
      if rank == 0:
          verboseprint(self.verbose, vars(self))

class FloquetMatrix:
    """Class that evaluates the Floquet Matrix of a time-periodically
       driven Bose Hubbard model
       Usage:
           HF = FloquetMatrix(p)
       
       Argument:
           p = An object instance of the ParamData class.

       Return value: 
           An object that stores an initiated PETSc Floquet Matrix 
    """  
    def __init__(self, params):
        """
        This creates a distributed parallel dense unit matrix
        """
        d = params.dimension
        #Setup the Floquet Matrix in parallel
        self.fmat = PETSc.Mat()
        self.fmat.create(comm=params.comm)
        self.fmat.setSizes([d,d])
        self.fmat.setType(PETSc.Mat.Type.DENSE)
        self.fmat.setUp()
        self.fmat.assemble()
        #Initialize it to unity
        diag = self.fmat.getDiagonal()
        diag.set(1.0)
        self.fmat.setDiagonal(diag)
        self.fmat.assemble()
    
    def evolve(self, params):
        """
        This evolves each column of the Floquet Matrix  in time via the 
        periodically driven Bose Hubbard model. The Floquet Matrix 
        is updated after one time period
        Usage:
            HF = FloquetMatrix(p)
            HF.evolve(p)
        Argument:
           p = An object instance of the ParamData class.
        Return value: 
           None
        TODO:
         petsc crashes. Need to debug basic petsc mat executions, then petsc4py
        """
        d = params.dimension
        times = np.linspace(0.0, 2.0 * np.pi/params.freq, num=timesteps)
        Istart, Iend = self.fmat.getOwnershipRange()
        for I in xrange(Istart, Iend):
            #Get the Ith row and evolve it
            (inds, dat) = self.fmat.getRow(I)
            #odeint does not handle complex numbers
            psi_t = \
                odeint(func,\
                        np.concatenate((dat[inds].real, dat[inds].imag)),\
                                              times, args=(params,), Dfun=None)
            #Set the Ith row to the final state after evolution  
            self.fmat.setValuesLocal(I,np.arange(d, dtype=petsc_int),\
                                            psi_t[-1][:d] + (1j)*psi_t[-1][d:])                                 
        self.fmat.assemble()

    def get_evals(self, params):
        """
        This diagonalizes the Floquet Matrix after evolution. Outputs the
        evals. It used PETSc/SLEPc to do this.
        
        Dense Floquet Matrices are being diagonalized without MPI using lapack
        So you can use multithreaded BLAS here, set by env vars
        Note that PETSc might have separate blas library links than 
        python/numpy.
        
        Usage:
            HF = FloquetMatrix(p)
            HF.evolve(p)
        Argument:
           p = An object instance of the ParamData class.
        Return value: 
           only root. tuple of array of eigenvalues and their errors
        TODO:
              Use petsc matload for large matrices that are petsc-dumped to disk 
        """
        rank = params.comm.Get_rank()            
        E = SLEPc.EPS() 
        E.create()
        E.setOperators(self.fmat)
        E.setType(SLEPc.EPS.Type.LAPACK)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        E.solve()
        nconv = E.getConverged()
        assert nconv==params.dimension, "All the eigenvalues failed to converge"
        if rank == 0:
            evals = np.zeros(nconv)
            evals_err = np.zeros(nconv)
        else:
            evals = None
            evals_err = None
        for i in xrange(nconv):
            #eigensys = E.getEigenpair(i, vr, vi)
            evals[i] = E.getEigenvalue(i)
            evals_err[i] = E.computeError(i)
        return (evals, evals_err)
    
if __name__ == '__main__':
  p = ParamData(lattice_size=3, particle_no=3, amp=1.0, freq=0.0, \
            int_strength=1.0, disorder_strength=1.0)
#!/usr/bin/env python
from __future__ import division, print_function

__docstring__ = """
Created on June 13 2016
@author: Analabha Roy (daneel@utexas.edu)

Schroedinger dynamics of the Bose Hubbard model with periodic drive (cosine):
    1. Floquet Matrix formulation and diagonalization
    2. Numerical time-evolution of initial condition 

To be run as as an imported module.

Usage:
    >>> import numpy as np
    >>> import sys
    >>> #Edit the path below to where you have kept the bose hubbard python module
    >>> bhpath = '/home/daneel/gitrepos/ising_exact'
    >>> sys.path.append(bhpath)
    >>> import bose_hubbard_exact_cosine as bh
    >>> from petsc4py import PETSc
    >>> Print = PETSc.Sys.Print
    >>> Print("These are the system parameters:")
    >>> p = bh.ParamData(lattice_size=3, particle_no=2, amp=0.1,\\
    >>>     freq=1.0, int_strength=1.0, disorder_strength=0.0, verbose=True)
    >>> Print("Hilbert Space Dimension = ",p.dimension)
    >>> Print("Initializing Floquet Matrix as unity:")
    >>> Hf = bh.FloquetMatrix(p)
    >>> Hf.fmat.view()
    >>> Print("Evolving the Floquet Matrix by one time period:")
    >>> Hf.evolve(p)
    >>> Print("New Floquet Matrix is the transpose of the below matrix:")
    >>> Hf.fmat.view()
    >>> Print("Transposing and diagonalizing the Floquet matrix. Eigenvalues:")
    >>> ev, err = Hf.tr_get_evals(p)
    >>> Print(ev)
    >>> Print("Modulii of eigenvalues:")
    >>> Print(np.abs(ev))

Note: 
    To get the above usage in a python script, just re-execute (in bash shell)
    and grep the python shell prompts
"""
import os, tempfile
import numpy as np
from math import factorial

from itertools import chain
from operator import sub

from bisect import bisect_left
from scipy.sparse import lil_matrix, dia_matrix
from scipy.integrate import odeint

from petsc4py import PETSc
Print = PETSc.Sys.Print
from slepc4py import SLEPc

timesteps = 100
petsc_int = np.int32 #petsc, by default. Uses 32-bit integers for indices
ode_dtype = np.float64 #scipy.odeint does not do complex numbers, so use this.
slepc_complex = np.complex128
fname = 'matrix-cache.dat' #Name of cached file for storing Floquet Matrix in disk

#Verbosity function
def verboseprint(verbosity, *args):
    if verbosity:
        for arg in args:
            Print(arg)
        Print(" ")

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
      return None    

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
        self.fmat = PETSc.Mat().create(comm=params.comm)
        self.fmat.setSizes([d,d])
        self.fmat.setType(PETSc.Mat.Type.DENSE)
        self.fmat.setUp()
        self.fmat.assemble()
        #Initialize it to unity
        diag = self.fmat.getDiagonal()
        diag.set(1.0)
        self.fmat.setDiagonal(diag)
        self.fmat.assemble()
        return None
    
    def evolve(self, params):
        """
        This evolves each ROW of the Floquet Matrix  in time via the 
        periodically driven Bose Hubbard model. The Floquet Matrix 
        is updated after one time period. To get the actual Floquet Matrix
        (which is supposed to be column ordered), always use PETSc to transpose
        it.
        Usage:
            HF = FloquetMatrix(p)
            HF.evolve(p)
        Argument:
           p = An object instance of the ParamData class.
        Return value: 
           None
        """
        d = params.dimension
        times = np.linspace(0.0, 2.0 * np.pi/params.freq, num=timesteps)
        rstart, rend = self.fmat.getOwnershipRange()
        #Storage for locally evaluated final state
        local_data = np.zeros((rend-rstart,d), dtype=slepc_complex)
        for loc_i, i in enumerate(xrange(rstart, rend)):
            #Get the Ith row and evolve it
            init = self.fmat[i,:]
            #Evolve to final state. Note:odeint does not handle complex numbers
            psi_t = \
                odeint(func,\
                        np.concatenate((init.real, init.imag)),\
                                              times, args=(params,), Dfun=None)         
            #Store the final state after evolution
            local_data[loc_i,:] = psi_t[-1][:d] + (1j)*psi_t[-1][d:]
        #Write the stored final states to the corresponding rows of fmat
        self.fmat[rstart:rend,:] = local_data
        self.fmat.assemble()
        return None
            
    def tr_get_evals(self, params, get_evecs=False, cachedir=None):
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
            HF.get_evals(p, disk_cache=True)
        Arguments:
            p            = An object instance of the ParamData class.
            get_evecs    = Boolean (optional, default False). Set to true
                            for getting the eigenvector matrix (row wise)
            cachedir     = Directory path as a string (optional, default None). 
                            If provided, then it is used to cache large Floquet 
                            matrices to temp file therein instead of in memory.
        Return value: 
            Tuple consisting of eigenvalues (array) and PETSc matrix of eigenvectors
        TODO:
              1. Use petsc matload for large matrices that are petsc-dumped to disk
              2. Add code to return both evals & evecs with evecs in parallel 
                 see routine in 'eth_diag.c' to see the logic for doing this
        """        
        rank = params.comm.Get_rank()    
        #Cache a large Floquet Matrix to disk
        if cachedir == None :
            cache = False
            cachefile = None
        else:
            cache = True
            assert type(cachedir) == str, "Please enter a valid path to cachedir"
            assert os.path.exists(cachedir),  "Please enter a valid path to cachedir"
            #Only root should create the file and broadcast to all proce
            cachefile = tempfile.NamedTemporaryFile(dir=cachedir) if rank == 0 else None
            fname = cachefile.name if rank == 0 else None
            params.comm.tompi4py().bcast(fname, root=0)
            #Now, initialize a PETSc viewer for this file, write and re-read
            #TODO: This does not work in parallel. Ask in petsc forum
            viewer = PETSc.Viewer().createBinary(fname, 'w')
            viewer.pushFormat(viewer.Format.NATIVE)
            viewer.view(self.fmat)
            viewer = PETSc.Viewer().createBinary(fname, 'r')
            self.fmat = PETSc.Mat().load(viewer)
        #Use SLEPc to diagonalize the matrix    
        E = SLEPc.EPS() 
        E.create()
        #Floquet Matrix was created as row ordered, but should be transpose
        #TODO: Please test this to make sure that the evec matrix is the
        #inverse of the original evec matrix
        E.setOperators(PETSc.Mat().createTranspose(self.fmat))
        E.setType(SLEPc.EPS.Type.LAPACK)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        E.solve()
        nconv = E.getConverged() #Number of converged eigenvalues
        assert nconv==params.dimension, "All the eigenvalues failed to converge"
        evals = np.empty(nconv, dtype=slepc_complex)
        evals_err = np.empty(nconv)
        if get_evecs:
            vr, wr = self.fmat.getVecs()
            vi, wi = self.fmat.getVecs()
        for i in xrange(nconv):
            evals[i] = E.getEigenpair(i, vr, vi) if get_evecs else E.getEigenvalue(i)
            evals_err[i] =  E.computeError(i)
        #Synchronize and have root close the cache file, if caching is done         
        params.comm.tompi4py().barrier()        
        if cache and rank == 0:
            cachefile.close()            
        return (evals, evals_err)
        
if __name__ == '__main__':
    Print(__docstring__)
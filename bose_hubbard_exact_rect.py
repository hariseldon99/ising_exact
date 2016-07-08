#!/usr/bin/env python
from __future__ import division, print_function
__doc__ = """
Created on July 02 2016
@author: Analabha Roy (daneel@utexas.edu)

Floquet Matrix formulation and diagonalization of the Bose Hubbard model with 
periodic drive (rectangular wave):

To be run as as an imported module.

Usage:
    >>> import numpy as np
    >>> import sys
    >>> #Edit the path below to where you have kept the bose hubbard python module
    >>> bhpath = '/path/to/bose_hubbard/module'
    >>> sys.path.append(bhpath)
    >>> import bose_hubbard_exact_rect as bh
    >>> from petsc4py import PETSc
    >>> Print = PETSc.Sys.Print
    >>> Print("These are the system parameters:")
    >>> p = bh.ParamData(lattice_size=3, particle_no=2, amp=0.1,\\
    >>>                               freq=1.0, int_strength=1.0, verbose=True)
    >>> Print("Hilbert Space Dimension = ",p.dimension)
    >>> Print("Initializing Floquet Matrix:")
    >>> Hf = bh.FloquetMatrix(p)
    >>> Hf.fmat.view()
    >>> Print("Evolving the Floquet Matrix by one time period:")
    >>> Hf.generate(p)
    >>> Print("New Floquet Matrix is the transpose of the below matrix:")
    >>> Hf.fmat.view()
    >>> Print("Transposing and diagonalizing the Floquet matrix. Eigenvalues:")
    >>> ev, err = Hf.eigensys(p)
    >>> ev.view()
    >>> Print("Modulii of eigenvalues:")
    >>> mods = ev.copy()
    >>> mods.sqrtabs()
    >>> mods.view()

Note: 
    To get the above usage in a python script, just re-execute (in bash shell)
    and grep the python shell prompts

References:
    [1] J M Zhang and R X Dong, Eur. Phys. J 31(3) 591 (2010). arXiv:1102.4006.
    [2] Penrose O and Onsager L, Phys. Ref. 104, 576-84 (1956).
"""

import os, tempfile as tmpf

import numpy as np
from math import factorial

from itertools import chain
from operator import sub

from bisect import bisect_left
from scipy.sparse import lil_matrix

from petsc4py import PETSc
Print = PETSc.Sys.Print
from slepc4py import SLEPc

timesteps = 100
petsc_int = np.int32 #petsc, by default. Uses 32-bit integers for indices
ode_dtype = np.float64 #scipy.odeint does not do complex numbers, so use this.
slepc_complex = np.complex128
slepc_mtol = 1e-7 #tolerance for matrix functions

def dumpclean(obj):
    """
    Neatly prints dictionaries line by line
    """
    if type(obj) == dict:
        for k, v in obj.items():
            if hasattr(v, '__iter__'):
                Print(k)
                dumpclean(v)
            else:
                Print('%s : %s' % (k, v))
    elif type(obj) == list:
        for v in obj:
            if hasattr(v, '__iter__'):
                dumpclean(v)
            else:
                Print(v)
    else:
        Print(obj)

def verboseprint(verbosity, *args):
    """
    print args if verbose mode is enabled
    """
    if verbosity:
        for arg in args:
            dumpclean(arg)
        Print(" ")

def vprint_eigen(verbosity, solver):
    """
    verbose print of an eigenvalue solver. also returns number of converged
    eigenvalues.
    """
    nconv = solver.getConverged() #Number of converged eigenvalues
    verboseprint(verbosity,"*** SLEPc Solution Results ***")
    its = solver.getIterationNumber()
    verboseprint(verbosity,"Number of iterations of the method: %d" % its)
    eps_type = solver.getType()
    verboseprint(verbosity,"Solution method: %s" % eps_type)
    nev, ncv, mpd = solver.getDimensions()
    verboseprint(verbosity,"Number of requested eigenvalues: %d" % nev)
    verboseprint(verbosity,"Number of converged eigenvalues: %d" % nconv)
    tol, maxit = solver.getTolerances()
    verboseprint(verbosity,"Stopping condition: tol=%.4g, maxit=%d" %\
    (tol, maxit))
    res = solver.getConvergedReason()
    verboseprint(verbosity,"Reason for convergence or divergence: %r"% res)
    verboseprint(verbosity,"Enumerated key of reason codes:")
    a =  vars(solver.ConvergedReason)
    reasons_key = {i:a[i] for i in a if 'CONVERGED' in i or 'DIVERGED' in i}
    verboseprint(verbosity, reasons_key)
    return nconv

class ParamData:
    """
    Class that stores Hamiltonian matrices and and system parameters. 
    """
    def __init__(self, lattice_size=3, particle_no=3, amp=1.0, freq=1.0, \
                 duty_cycle=0.5, int_strength=1.0, field=None,lapack=False,\
                                     mpicomm=PETSc.COMM_WORLD, verbose=False):                       
      """
       Usage:
       p = ParamData(lattice_size=3, particle_no=3, amp=1.0, freq=0.0, \
            int_strength=1.0, disorder_strength=1.0, mpicomm=comm)
		      
       All parameters (arguments) are optional.
       
       Parameters:
       lattice_size = The size of your lattice as an integer.
       particle_no  = The number of particles (bosons) in the lattice 
       amp          = The periodic (rectangle) drive amplitude. Defaults to 1.
       freq   	    = The periodic (rectangle) drive frequency. Defaults to 1.
       duty_cycle   = The duty cycle of the rectangular wave. Betwn 0-1. 
                       Defaults to 0.5 (square wave) 
       int_strength = The interaction strength U of the Bose Hubbard model.
                       Defaults to 1.
       field        = Optional numpy array of spatially varying field. 
                       Defaults to None.
       lapack       = Boolean. Default False. Use serial lapack to diagonalize 
                       the hamiltonian.   
       mpicomm      = MPI Communicator. Defaults to PETSc.COMM_WORLD
       verbose      = Boolean for verbose output. Defaults to False

       Return value: 
       An object that stores all the parameters above, as well as the fock states
       and the kinetic and potential energy matrices. Also ground state stored
       as a petsc vec. Finally, the matrix elements of the off-diagonal order 
       parameter a^\dagger_0 a_[M/2] (M = lattice_size) are stored.
      """
      self.lattice_size = lattice_size
      self.particle_no = particle_no
      self.amp, self.freq = amp, freq
      assert 0. <= duty_cycle <= 1. , "Please enter a duty cycle between 0 and 1"
      self.duty_cycle = duty_cycle
      self.int_strength = int_strength
      self.comm = mpicomm
      self.verbose = verbose
      n,m = self.particle_no, self.lattice_size    
      rank = self.comm.Get_rank()
      if field is not None:
          assert type(field).__module__ == np.__name__, "Field needs a numpy array"
          assert field.size >= lattice_size, "Field array too small"
          h = field.flatten() if rank == 0 else None
          h = self.comm.tompi4py().bcast(h, root=0) #sync field vals w root
          self.field = True
      else:
          self.field = False
          h = 0.0
      #Dimensionality of the hilbert space
      self.dimension = np.int(factorial(n+m-1)/(factorial(n) * factorial(m-1)))
      size = self.comm.Get_size()      
      assert self.dimension >= size, "There are fewer rows than MPI procs!"
      d = self.dimension
      if rank == 0:
          verboseprint(self.verbose, vars(self))
      self.tagweights = np.sqrt(100 * np.arange(m) + 3)
      if rank == 0:
          #Build the dxm-size fock state matrix as per ref[1] in docstring
          self.all_fockstates =  lil_matrix((d, m), dtype=ode_dtype)
          self.fockstate_tags = np.zeros(d)
          for row_i, row in enumerate(self.boxings(n,m)):
              self.all_fockstates[row_i,:] = row
              row = np.array(row)
              self.fockstate_tags[row_i] = self.tagweights.dot(row)
          #Sort the tags and store the original indices
          self.tag_inds = np.argsort(self.fockstate_tags)
          self.fockstate_tags = self.fockstate_tags[self.tag_inds]
      else:
          self.all_fockstates = None          
          self.tag_inds = None          
          self.fockstate_tags = None
      self.all_fockstates = self.comm.tompi4py().bcast(self.all_fockstates, root=0)        
      self.tag_inds = self.comm.tompi4py().bcast(self.tag_inds, root=0)    
      self.fockstate_tags = self.comm.tompi4py().bcast(self.fockstate_tags, root=0)    
      #Build the interaction + field part of the hamiltonian
      self.hamilt_int = PETSc.Mat().create(comm=self.comm)
      self.hamilt_int.setSizes([d,d])
      self.hamilt_int.setUp() 
      rstart, rend = self.hamilt_int.getOwnershipRange()
      self.int_matset(self.hamilt_int, (rstart, rend),h)
      self.hamilt_int.assemble()
      #Now, set the kinetic energy matrix separately
      self.hamilt_ke = self.hamilt_int.duplicate()
      rstart, rend = self.hamilt_ke.getOwnershipRange()
      for site in xrange(m):  
              next_site = 0 if site==m-1 else site+1 #Periodic bc
              self.ke_matset(self.hamilt_ke, site, next_site, (rstart, rend))
      self.hamilt_ke.assemble()
      #Now, set the off diagonal order parameter matrix elements
      self.offd_order = self.hamilt_int.duplicate()
      rstart, rend = self.offd_order.getOwnershipRange()
      site, next_site = 0, np.floor(self.lattice_size/2.)
      self.ke_matset(self.offd_order, site, next_site, (rstart, rend))
      self.offd_order.assemble()
      #Now, get the ground state ONLY using SLEPc
      E = SLEPc.EPS() 
      E.create()
      E.setOperators(self.hamilt_ke + self.hamilt_int)
      if lapack:
          E.setType(SLEPc.EPS.Type.LAPACK)
      E.setProblemType(SLEPc.EPS.ProblemType.HEP)
      E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
      E.solve()
      nconv = vprint_eigen(self.verbose, E)
      assert nconv >= 1, "Convergence of ground state eigensolver failed!!!"
      self.groundstate, wr = self.hamilt_ke.getVecs()
      vi, wi = self.hamilt_ke.getVecs()
      self.groundstate_energy = E.getEigenpair(0, self.groundstate, vi)
      #Complexify the eigenvector by vr = (1j)* vi + vr
      self.groundstate.axpy(1j,vi)           

    def ke_matset(self, mat, i, j, local_block):
          """
          Sets all the matrix elements of kinetic energy 'mat' in the fock state
          representation that come from the lattice sites 'i' and 'j'
          """               
          i,j = int(i), int(j)                                                  
          for v, fock_v in enumerate(self.all_fockstates):
              fock_v = fock_v.toarray().flatten()
              #Note that, if any site has no particles, then one annihilation
              #operator nullifies the whole state
              if fock_v[j] != 0:
                  #Hop a particle in fock_v from i to i+1 & store.
                  fockv_hopped = np.copy(fock_v)
                  fockv_hopped[i] += 1
                  fockv_hopped[j] -= 1
                  #Tag this hopped vector
                  tag = self.tagweights.dot(fockv_hopped)
                  #Binary search for this tag in the sorted array of tags
                  w = self.index(self.fockstate_tags, tag) 
                  #Get the corresponding row index 
                  u = self.tag_inds[w]
                  rstart, rend = local_block
                  if rstart <= u < rend: #set local row val only
                      val = -np.sqrt((fock_v[i]+1) * fock_v[j])
                      mat[u,v] = val
                      mat[v,u] = val
     
    def int_matset(self, mat, local_block, field):
        U = self.int_strength  
        for v, fock_v in enumerate(self.all_fockstates):
            (rstart, rend) = local_block    
            if rstart <= v < rend: #set locally
              fockv_den = fock_v.toarray()
              mat[v,v] = np.sum(fockv_den * (U * (fockv_den-1) - field))
    
    def orderparam(self, state):
        """
        This gets the order parameter of the state vector provided
        i.e, the expectation value of the offd_order matrix in ParamData.
        See "Penrose Onsager criterion" [2].
        """
        dummy, new_vector =  self.offd_order.getVecs()  
        self.offd_order.mult(state, new_vector)
        return np.abs(new_vector.dot(state))

    def index(self, a, x):
        """
        Locate the leftmost value exactly equal to x
        """
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        raise ValueError

    def boxings(self, n, k):
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

class FloquetMatrix:
    
    def __init__(self, params):
        """
        Class that evaluates the Floquet Matrix of a time-periodically
        driven Bose Hubbard model
        Usage:
            HF = FloquetMatrix(p)
       
        Argument:
            p = An object instance of the ParamData class.

        Return value: 
            An object that stores an initiated PETSc Floquet Matrix 
        """
        pass        
    
    def solve_exp(self, t, H, b, x):
        """
        Setup the and solve the petsc matrix function |x> = exp(-IHt)|b>
        See SLEPc manual chapters 7 and 8 to understand this
        """
        s = -1j * t
        if self.isdiag(H): #If H is diagonal, then exponentiation is trivial
            x = H.getDiagonal()
            x.scale(s)
            x.exp()
            x.pointwiseMult(x,b)
        else: #This method fails for diagonal matrices. Krylov don't work 
            M = SLEPc.MFN().create()
            M.setOperator(H)
            f = M.getFN()
            f.setType(SLEPc.FN.Type.EXP)
            f.setScale(s)
            M.setTolerances(slepc_mtol)
            M.solve(b,x)
            
    def solve_pow(self, n, H, b, x):
        """
        Setup the and solve the petsc matrix function |x> = H^n|b>
        See SLEPc manual chapters 7 and 8 to understand this
        """
        if self.isdiag(H): #If H is diagonal, then power is trivial
            x = H.getDiagonal()
            for i in xrange(1,n): #pointwisemult n-1 times
                x.pointwiseMult(x,x)
            x.pointwiseMult(x,b)
        else: #This method fails for diagonal matrices. Krylov don't work    
            M = SLEPc.MFN().create()
            M.setOperator(H)
            f = M.getFN()
            f.setType(SLEPc.FN.Type.RATIONAL)
            f.setRationalDenominator([1])
            f.setRationalNumerator(np.eye(1,n+1,n).flatten()[::-1])
            M.setTolerances(slepc_mtol)
            M.solve(b,x)

    def isdiag(self, A):
        """
        Checks PETSc matrix to determine if diag. It stores diag, zeros it and 
        calcs the inf-norm/size. If small, then matrix sans diag is 0. Then reset.
        """
        size = A.getSize()
        D = min(size)
        diag = A.getDiagonal()
        zero = diag.duplicate()
        zero.set(0.0)
        A.setDiagonal(zero)
        A.assemble()
        n =  A.norm(norm_type=PETSc.NormType.INF)/D
        is_diag = True if n <= slepc_mtol else False
        A.setDiagonal(diag)
        A.assemble()
        return is_diag
        
    def ralloc(self, mat, row, vec):
        """
        Allocate the data in petsc vec to the row in petsc mat
        """
        locfirst, loclast = vec.getOwnershipRange()
        loc_idx = range(locfirst, loclast)
        dataloc = vec.getArray()
        mat.setValues([row],loc_idx,dataloc)
        
    def generate(self, params):
        """
        This evaluates the Floquet Matrix of rectangle wave drive
        If the Hamiltonian is H_1 for the first cycle of the period  
        and H_2 for the second cycle of the period. Then Floquet
        Matrix is exp(-I H_2 (T-t)) \times exp(-I H_1 t).
        Usage:
            HF = FloquetMatrix(p)
            HF.generate(p)
            Hf.fmat.view()
        Argument:
           p = An object instance of the ParamData class.
        """
        soln, unit = params.hamilt_ke.getVecs()
        T = 2.0 * np.pi/params.freq
        t1 = params.duty_cycle * T
        hamilt_1 = (1.0 + params.amp) * params.hamilt_ke + params.hamilt_int
        fmat_1 = PETSc.Mat().create()
        fmat_1.setSizes([params.dimension, params.dimension])
        fmat_1.setUp()
        fmat_1.assemble()
        t2 = (1.0 - params.duty_cycle) * T
        hamilt_2 = (1.0 - params.amp) * params.hamilt_ke + params.hamilt_int
        fmat_2 = fmat_1.duplicate()
        for col in xrange(params.dimension):
            unit.set(0)
            unit[col] = 1.0
            unit.assemble()
            self.solve_exp(t1, hamilt_1, unit, soln)
            self.ralloc(fmat_1, col, soln)
            self.solve_exp(t2, hamilt_2, unit, soln)
            self.ralloc(fmat_2, col, soln)
        fmat_1.assemble()
        fmat_2.assemble()
        #Transpose to column ordered floquet matrices
        fmat_1.transpose()
        fmat_2.transpose()
        self.fmat = fmat_2.matMult(fmat_1)
        
    def evolve(self, time, params):
        """
        This evolves the ground state by integer multiples of the time period.
        It raises the Floquet matrix to the nth power and multiplies to the state.
        
        Usage:
            HF = FloquetMatrix(p)
            HF.generate(p)
            state = HF.evolve(10,p)
        Arguments:
            time      = This needs to be an integer. The state will be evolved
                        in integer multiples of the time period i.e 2*pi/p.freq
            p         = An object instance of the ParamData class.
            
        Return value:
            PETSc vec of the final state
        """
        #We want to compute self.fmat^time |self.groundstate>
        assert isinstance( time , ( int, long ) ), "Time needs to be an integer"
        dummy, final_state = self.fmat.getVecs()
        self.solve_pow(time, self.fmat, params.groundstate, final_state)
        return final_state
        
    def eigensys(self, params, get_evecs=False, cachedir=None, lapack=False):
        """
        This diagonalizes the Floquet Matrix after evolution. Outputs the
        evals. It used PETSc/SLEPc to do this.
        
        Usage:
            HF = FloquetMatrix(p)
            HF.evolve(p)
            HF.eigensys(p, disk_cache=True)
        Arguments:
            p            = An object instance of the ParamData class.
            get_evecs    = Boolean (optional, default False). Set to true
                            for getting the eigenvector matrix (row wise)                
            cachedir     = Directory path as a string (optional, default None). 
                            If provided, then it is used to cache large Floquet 
                            matrices to temp file in there instead of memory.
            lapack       = Boolean. Default False. Use serial lapack to 
                            diagonalize the Floquet matrix.
        Return value: 
            Tuple consisting of eigenvalues (array) and their errors both as
            PETSc vectors. If evaluated, the eigenvector matrix is stored as 
            PETSc Mat COLUMN WISE in "HF.evecs"
        """     
        #Initiate the SLEPc solver to diagonalize the matrix    
        E = SLEPc.EPS() 
        E.create()
        #Now, cache large Floquet matrix to disk before passing it to the solver
        rank = params.comm.Get_rank()    
        if cachedir == None :
            cache = False
            cfile = None
            fmat_loc = self.fmat
        else:
            cache = True
            assert type(cachedir) == str, "Please enter a valid path to cachedir"
            assert os.path.exists(cachedir),\
                                        "Please enter a valid path to cachedir"
            #Only root should create the file and broadcast to all proce
            cfile = tmpf.NamedTemporaryFile(dir=cachedir) if rank == 0 else None
            fname = cfile.name if rank == 0 else None
            fname = params.comm.tompi4py().bcast(fname, root=0)
            #Now, initialize a PETSc viewer for this file, write and re-read
            viewer = PETSc.Viewer().createBinary(fname, 'w')
            viewer.pushFormat(viewer.Format.NATIVE)
            viewer.view(self.fmat)
            viewer = PETSc.Viewer().createBinary(fname, 'r')    
            fmat_loc = self.fmat.duplicate()
            fmat_loc.load(viewer)
        #Finish setting up the eigensolver and execute it
        E.setOperators(fmat_loc)
        if lapack:
            E.setType(SLEPc.EPS.Type.LAPACK)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        E.setDimensions(nev=params.dimension)#Need all the eigenvalues
        E.solve()
        nconv = vprint_eigen(params.verbose, E)
        assert nconv == params.dimension,\
            "Only %r out of %r eigenvalues converged" % (nconv, params.dimension)
        evals, er = fmat_loc.getVecs()
        evals_err, eer = fmat_loc.getVecs()
        if get_evecs:
            self.evecs = fmat_loc.duplicate()
            vr, wr = fmat_loc.getVecs()
            vi, wi = fmat_loc.getVecs()
        for i in xrange(nconv):
            ev = E.getEigenpair(i, vr, vi) if get_evecs else E.getEigenvalue(i)
            ev_err =  E.computeError(i)
            evals.setValue(i,ev)
            evals_err.setValue(i,ev_err)
            if get_evecs:
                #Complexify the eigenvector by vr = (1j)* vi + vr
                vr.axpy(1j,vi)
                self.ralloc(self.evecs, i, vr)
        evals.assemble()
        evals_err.assemble()
        if get_evecs:
            self.evecs.assemble()
            self.evecs.transpose() #For column ordering
        #Synchronize and have root close the cache file, if caching is done         
        params.comm.tompi4py().barrier()        
        if cache and rank == 0:
            cfile.close()            
        return (evals, evals_err)
        
if __name__ == '__main__':
    Print(__doc__)
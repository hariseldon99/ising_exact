#!/usr/bin/env python
"""
Created on May  14 2015

@author: Analabha Roy (daneel@utexas.edu)

Usage : ./ising_exact_sparse.py -h
"""
import sys, argparse
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pprint import pprint
from itertools import combinations

"""
Default Parameters are entered here
"""
#Lattice size
L = 6
t_init = 0.0 # Initial time
t_final = 60.0 # Final time
n_steps = 1000 # Number of time steps

#Power law decay of interactions
beta = 1.0

desc = """Dynamics by exact diagonalization of 
		1d Ising model with long range interactions"""

#Pauli matrices
sig_x, sig_y, sig_z = \
  np.array([[0j, 1.0+0j], [1.0+0j, 0j]]), \
    np.array([[0j, -1j], [1j, 0j]]), \
      np.array([[1.0+0j, 0j], [0j, -1+0j]])  

#Sparsify them for sparse kronecker products
sig_x, sig_y, sig_z = \
  sparse.csr_matrix(sig_x), \
    sparse.csr_matrix(sig_y), \
      sparse.csr_matrix(sig_z)

def input():
  parser = argparse.ArgumentParser(description=desc)
  
  parser.add_argument('-l', '--lattice_size', type=np.int64,\
    help="size of the lattice", default=L)
  
  parser.add_argument('-omx', '--output_magx',\
    help="sx (magnetization) output file", default="sx_outfile.txt")
  parser.add_argument('-omy', '--output_magy',\
    help="sy (magnetization) output file", default="sy_outfile.txt")
  parser.add_argument('-omz', '--output_magz',\
    help="sz (magnetization) output file", default="sz_outfile.txt")

  parser.add_argument('-ox', '--output_sxvar',\
    help="sx variance output file", default="sxvar_outfile.txt")
  parser.add_argument('-oy', '--output_syvar',\
    help="sy variance output file", default="syvar_outfile.txt")
  parser.add_argument('-oz', '--output_szvar',\
    help="sz variance output file", default="szvar_outfile.txt")

  parser.add_argument('-oxy', '--output_sxyvar',\
    help="sxy variance output file", default="sxyvar_outfile.txt")
  parser.add_argument('-oxz', '--output_sxzvar',\
    help="sxz variance output file", default="sxzvar_outfile.txt")
  parser.add_argument('-oyz', '--output_syzvar',\
    help="sx variance output file", default="syzvar_outfile.txt")

  parser.add_argument("-v", '--verbose', action='store_true', \
    help="increase output verbosity")

  parser.add_argument('-pbc', '--periodic', \
    help="switch to periodic boundary conditions, default is open",\
      action="store_true")
  parser.add_argument('-n', '--nonorm', \
    help="do not normalize the energy per particle, default is yes",\
      action="store_true")
  
  parser.add_argument('-b', '--beta', type=np.float64, \
    help="power law index for long range interactions", default=beta)

  parser.add_argument('-x', '--hx', type=np.float64, \
    help="x transverse field", default=1.0)
  parser.add_argument('-y', '--hy', type=np.float64, \
    help="y transverse field", default=0.0)
  parser.add_argument('-z', '--hz', type=np.float64, \
    help="z transverse field", default=0.0)
  parser.add_argument('-jx', '--jx', type=np.float64, \
    help="x hopping", default=0.0)
  parser.add_argument('-jy', '--jy', type=np.float64, \
    help="y hopping", default=0.0)
  parser.add_argument('-jz', '--jz', type=np.float64, \
    help="z hopping", default=1.0)

  return parser.parse_args()

#This is the Jmn hopping matrix with power law decay for open boundary
#conditions.
def get_jmat_obc(args):
  lsize = args.lattice_size
  #Set the diagonal bands of the matrix
  J = np.sum(np.diagflat(np.ones(lsize-i)*pow(i, -args.beta), i) \
    for i in xrange(1, lsize))
  J = J+J.T
  np.fill_diagonal(J, 0)
  return J

#This is the Jmn hopping matrix with power law decay for periodic boundary
#conditions.
def get_jmat_pbc(args):
  lsize = args.lattice_size
  mid_diag = np.floor(lsize/2).astype(int)
  #Set the diagonal bands of the matrix
  J = np.sum(np.diagflat(np.ones(lsize-i)*pow(i, -args.beta), i) \
    for i in xrange(1, mid_diag+1))+\
      np.sum(np.diagflat(np.ones(lsize-i)*pow(lsize-i, -args.beta), i)\
	for i in xrange(mid_diag+1, lsize))
  J = J + J.T
  np.fill_diagonal(J, 0)
  return J

class ParamData:
  description = """Class to store parameters and hopping matrix"""
  def __init__(self, args):
      #Copy arguments from parser to this class
      self.__dict__.update(args.__dict__)
      self.jvec = np.array([args.jx, args.jy, args.jz])
      self.hvec = np.array([args.hx, args.hy, args.hz])
      lsize = self.lattice_size
      if args.periodic:
	  self.periodic_boundary_conditions = 1
	  self.open_boundary_conditions = 0
	  self.jmat = get_jmat_pbc(args)
	  mid = np.floor(lsize/2).astype(int)
	  self.norm =\
	    2.0 * np.sum(1/(pow(np.arange(1, mid+1), args.beta).astype(float)))
      else:
	  self.periodic_boundary_conditions = 0
	  self.open_boundary_conditions = 1
	  self.jmat = get_jmat_obc(args)
	  self.norm =\
	    np.sum(1/(pow(np.arange(1, lsize+1), args.beta).astype(float)))
      if args.nonorm:
	  self.norm = 1.0      

class Hamiltonian:
  description = """Precalculates all the dynamics information 
		   of the Hamiltonian""" 	  
  
  def nummats(self, mu):
	lsize = self.lattice_size
	#Left Hand Side
	if(mu == 0):
	  num_x, num_y, num_z  = sig_x, sig_y, sig_z
	else:
	  id = sparse.identity(2**mu, format='csr')
	  num_x, num_y, num_z  = \
	   sparse.kron(id,sig_x),sparse.kron(id,sig_y),sparse.kron(id,sig_z)
	#Right Hand Side    
	if(mu < lsize - 1):
	    id = np.eye(2**(lsize-mu-1),2**(lsize-mu-1))
	    num_x, num_y, num_z = \
	     sparse.kron(num_x, id),sparse.kron(num_y, id),sparse.kron(num_z, id)    
	return (num_x, num_y, num_z)
      
  def kemats(self, sitepair):
    lsize = self.lattice_size
    (mu, nu) = sitepair
    #Left Hand Side    
    if(mu == 0):
	ke_x, ke_y, ke_z = sig_x, sig_y, sig_z
    else:
	id = sparse.identity(2**mu, format='csr')
	ke_x, ke_y, ke_z = \
	 sparse.kron(id, sig_x),sparse.kron(id, sig_y),sparse.kron(id, sig_z)
    #Middle Side
    dim = 1 if mu == nu else 2**(np.abs(mu-nu)-1) 
    id = sparse.identity(dim, format='csr')
    ke_x, ke_y, ke_z = \
     sparse.kron(ke_x, id),sparse.kron(ke_y, id),sparse.kron(ke_z, id)
    ke_x, ke_y, ke_z = \
     sparse.kron(ke_x, sig_x),sparse.kron(ke_y, sig_y),sparse.kron(ke_z, sig_z) 
    #Right Hand Side    
    if(nu < lsize - 1):
      id = sparse.identity(2**(lsize-nu-1), format='csr')
      ke_x, ke_y, ke_z = \
     sparse.kron(ke_x, id),sparse.kron(ke_y, id),sparse.kron(ke_z, id)
    return (ke_x, ke_y, ke_z) 
  
  def offd_corrmats(self, sitepair):
    lsize = self.lattice_size
    (mu, nu) = sitepair
    #Left Hand Side    
    if(mu == 0):
	cxy, cxz, cyz = sig_x, sig_x, sig_y 
    else:
	id = sparse.identity(2**mu, format='csr')
	cxy, cxz, cyz = \
	 sparse.kron(id, sig_x),sparse.kron(id, sig_x),sparse.kron(id, sig_y)
    #Middle Side
    dim = 1 if mu == nu else 2**(np.abs(mu-nu)-1) 
    id = sparse.identity(dim, format='csr')
    cxy, cxz, cyz = \
     sparse.kron(cxy, id),sparse.kron(cxz, id),sparse.kron(cyz, id)
    cxy, cxz, cyz = \
     sparse.kron(cxy, sig_y),sparse.kron(cxz, sig_z),sparse.kron(cyz, sig_z) 
    #Right Hand Side    
    if(nu < self.lattice_size - 1):
      id = sparse.identity(2**(lsize-nu-1), format='csr')
      cxy, cxz, cyz = \
     sparse.kron(cxy, id),sparse.kron(cxz, id),sparse.kron(cyz, id)
    return (cxy, cxz, cyz) 
  
  def __init__(self, params):
    #Copy arguments from params to this class
    self.__dict__.update(params.__dict__)
    #Build KE matrix & convert to numpy array for full diagonalization
    H = self.jx * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[0] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0).toarray()
    H += self.jy * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[1] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0).toarray()
    H += self.jz * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[2] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0).toarray()
    H = H/self.norm
    #Build transverse field matrix
    H += self.hx * np.sum(np.array([self.nummats(mu)[0] \
      for mu in xrange(self.lattice_size)]), axis=0).toarray()
    H += self.hy * np.sum(np.array([self.nummats(mu)[1] \
      for mu in xrange(self.lattice_size)]), axis=0).toarray()
    H += self.hz * np.sum(np.array([self.nummats(mu)[2] \
      for mu in xrange(self.lattice_size)]), axis=0).toarray()
    try:
	evals, U = np.linalg.eigh(-H)
	idx = evals.argsort()
	evals = evals[idx]
	U = U[:,idx]
	self.esys = (evals, U)
    except np.linalg.linalg.LinAlgError:
	self.esys = None
	
  def evolve(self, obs, times, state):
    #Given the eigensystem of time-translation,
    #build the translation and evolve an obs
    if self.esys is not None:
      Z = sparse.csr_matrix(self.esys[1])
      list_of_time_obs_data = []
      for t in times:
	Hdt = sparse.diags(np.exp( (1j) * self.esys[0] * t), 0, \
	  format='csr')
	obs_t = Z.T * Hdt * Z * obs * Z.T * Hdt.T.conjugate() * Z
        list_of_time_obs_data.append(np.vdot(\
	  state, np.dot(obs_t.toarray(), state)))
      return np.array(list_of_time_obs_data)
    else:
      return None
  
class OutData:
  description = """Class to store output data"""
  def __init__(self, t, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz, params):
      self.t_output = t
      self.sx, self.sy, self.sz = sx, sy, sz
      self.sxvar, self.syvar, self.szvar = sxx, syy, szz
      self.sxyvar, self.sxzvar, self.syzvar = sxy, sxz, syz
      self.__dict__.update(params.__dict__)

  def dump_data(self):
      np.savetxt(self.output_magx, np.vstack((self.t_output, self.sx)).T,\
	delimiter=' ')
      np.savetxt(self.output_magy, np.vstack((self.t_output, self.sy)).T,
		  delimiter=' ')
      np.savetxt(self.output_magz, np.vstack((self.t_output, self.sz)).T,\
	delimiter=' ')
      np.savetxt(self.output_sxvar, \
	np.vstack((self.t_output, self.sxvar)).T, delimiter=' ')
      np.savetxt(self.output_syvar, \
	np.vstack((self.t_output, self.syvar)).T, delimiter=' ')
      np.savetxt(self.output_szvar, \
	np.vstack((self.t_output, self.szvar)).T, delimiter=' ')
      np.savetxt(self.output_sxyvar, \
	np.vstack((self.t_output, self.sxyvar)).T, delimiter=' ')
      np.savetxt(self.output_sxzvar, \
	np.vstack((self.t_output, self.sxzvar)).T, delimiter=' ')
      np.savetxt(self.output_syzvar, \
	np.vstack((self.t_output, self.syzvar)).T, delimiter=' ')
     
def runising_dyn(params):
  if params.verbose:
    print "Executing diagonalization with parameters:"
    pprint(vars(params), depth=1)
  else:
    print "Starting run ..."
 
  h = Hamiltonian(params)  
  
  if h.esys is not None:
    
    lsize = h.lattice_size
    lsq = lsize * lsize
        
    sx = np.sum(np.array([h.nummats(mu)[0] \
      for mu in xrange(h.lattice_size)]), axis=0)
    sy = np.sum(np.array([h.nummats(mu)[1] \
      for mu in xrange(h.lattice_size)]), axis=0)
    sz = np.sum(np.array([h.nummats(mu)[2] \
      for mu in xrange(h.lattice_size)]), axis=0)
    sx, sy, sz = sx/lsize, sy/lsize, sz/lsize
      
    offset = (1/lsize) * \
      sparse.identity(2**lsize, format='csr', dtype=complex)
    
    sxvar = np.sum(np.array( [h.jmat[sitepair] * h.kemats(sitepair)[0] \
      for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    syvar = np.sum(np.array( [h.jmat[sitepair] * h.kemats(sitepair)[1] \
      for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    szvar = np.sum(np.array( [h.jmat[sitepair] * h.kemats(sitepair)[2] \
      for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    sxvar, syvar, szvar = \
      (sxvar/lsq) + offset, (syvar/lsq) + offset, (szvar/lsq) + offset
    
    sxyvar = np.sum(np.array(\
      [h.jmat[sitepair] * h.offd_corrmats(sitepair)[0] \
	for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    sxzvar = np.sum(np.array(\
      [h.jmat[sitepair] * h.offd_corrmats(sitepair)[1] \
	for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    syzvar = np.sum(np.array(\
      [h.jmat[sitepair] * h.offd_corrmats(sitepair)[2] \
	for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
    sxyvar, sxzvar, syzvar = \
      (sxyvar/lsq) + offset, (sxzvar/lsq) + offset, (syzvar/lsq) + offset
        
    #Now, the expectation value of any observable A is just
    # U^\dagger(t) A U(t). In the canonical basis, U(t) is just
    #\sum_n e^{i e_n t}|n><n| where e_n are the eigenvals!
    #Assume that psi_0 is the eigenstate of \sum_\mu sigma^x_\mu
    initstate =  np.ones(2**lsize)/np.sqrt(2**lsize)
     
    dt = (t_final-t_init)/(n_steps-1.0)
    t_output = np.arange(t_init, t_final, dt)
    
    sxdata = h.evolve(sx, t_output, initstate)
    sydata = h.evolve(sy, t_output, initstate)
    szdata = h.evolve(sz, t_output, initstate)
    
    sxvar_data = h.evolve(sxvar, t_output, initstate)
    sxvar_data = sxvar_data - (sxdata)**2
    
    syvar_data = h.evolve(syvar, t_output, initstate)
    syvar_data = syvar_data - (sydata)**2
    
    szvar_data = h.evolve(szvar, t_output, initstate)
    szvar_data = szvar_data - (szdata)**2
    
    sxyvar_data = h.evolve(sxyvar, t_output, initstate)
    sxyvar_data = sxyvar_data - (sxdata) * (sydata)
    
    sxzvar_data = h.evolve(sxzvar, t_output, initstate)
    sxzvar_data = sxzvar_data - (sxdata) * (szdata)
    
    syzvar_data = h.evolve(syzvar, t_output, initstate)
    syzvar_data = syzvar_data - (sydata) * (szdata)
    
    data = OutData(t_output, np.abs(sxdata), np.abs(sydata), \
      np.abs(szdata), np.abs(sxvar_data), np.abs(syvar_data), \
    	np.abs(szvar_data), np.abs(sxyvar_data), \
    	      np.abs(sxzvar_data), np.abs(syzvar_data), params)
    
    if params.verbose:
      #Plot the eigenvalues and eigenvectors
      lsize = params.lattice_size
      limits = (-2**lsize/100,2**lsize)
      plt.plot(h.esys[0].real)
      plt.title('Eigenvalues - Sorted')
      plt.matshow(np.abs(h.esys[1])**2,interpolation='nearest',\
      cmap=cm.coolwarm)
      plt.title('Eigenvectors - Sorted' ) 
      plt.colorbar()
      fig, ax = plt.subplots()
      diag_cc = np.abs(h.esys[1].dot(initstate))**2
      print diag_cc
      plt.bar(np.arange(2**lsize), diag_cc, edgecolor='blue')                
      plt.xlim(limits)
      plt.xscale('log')
      ax.xaxis.tick_top()
      ax.set_xlabel('Lexicographical order of eigenstate')
      ax.xaxis.set_label_position('top') 
      ax.set_ylabel('Initial state probability wrt eigenstate')            
      plt.show()
      
    print "\nDumping outputs to files ..."
    data.dump_data()
    print 'Done'
    
  else:
    print "Error! Eigenvalues did not converge for these parameters,\
    skipping ..."

if __name__ == '__main__':
    args_in = input()
    paramdat = ParamData(args_in)
    runising_dyn(paramdat)

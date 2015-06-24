#!/usr/bin/env python
"""
Created on May  14 2015

@author: Analabha Roy (daneel@utexas.edu)

Usage : ./ising_exact.py -h
"""
import sys, argparse
import numpy as np
from pprint import pprint
from itertools import combinations
from scipy.integrate import odeint
from scipy.sparse import *

"""
Default Parameters are entered here
"""
#Lattice size
L = 6
t_init = 0.0 # Initial time
t_final = 4.0 # Final time
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
    help="x transverse field", default=0.0)
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
  N = args.lattice_size
  J = dia_matrix((N, N))
  for i in xrange(1,N):
    elem = pow(i, -args.beta)
    J.setdiag(elem, k=i)
    J.setdiag(elem, k=-i)
  return J.toarray()

#This is the Jmn hopping matrix with power law decay for periodic boundary
#conditions.
def get_jmat_pbc(args):
  N = args.lattice_size
  J = dia_matrix((N, N))
  mid_diag = np.floor(N/2).astype(int)
  for i in xrange(1,mid_diag+1):
    elem = pow(i, -args.beta)
    J.setdiag(elem, k=i)
    J.setdiag(elem, k=-i)
  for i in xrange(mid_diag+1, N):
    elem = pow(N-i, -args.beta)
    J.setdiag(elem, k=i)
    J.setdiag(elem, k=-i)
  return J.toarray()  

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
	  id = np.eye(2**mu,2**mu)
	  num_x, num_y, num_z  = \
	    np.kron(id,sig_x), np.kron(id,sig_y), np.kron(id,sig_z)
	#Right Hand Side    
	if(mu < lsize - 1):
	    id = np.eye(2**(lsize-mu-1),2**(lsize-mu-1))
	    num_x, num_y, num_z = \
	      np.kron(num_x, id), np.kron(num_y, id), np.kron(num_z, id)    
	return (num_x, num_y, num_z)
      
  def kemats(self, sitepair):
    lsize = self.lattice_size
    (mu, nu) = sitepair
    #Left Hand Side    
    if(mu == 0):
	ke_x, ke_y, ke_z = sig_x, sig_y, sig_z
    else:
	id = np.eye(2**mu,2**mu)
	ke_x, ke_y, ke_z = \
	  np.kron(id, sig_x), np.kron(id, sig_y), np.kron(id, sig_z)
    #Middle Side
    dim = 1 if mu == nu else 2**(np.abs(mu-nu)-1) 
    id = np.eye(dim,dim)
    ke_x, ke_y, ke_z = \
      np.kron(ke_x, id), np.kron(ke_y, id), np.kron(ke_z, id)
    ke_x, ke_y, ke_z = \
      np.kron(ke_x, sig_x), np.kron(ke_y, sig_y), np.kron(ke_z, sig_z) 
    #Right Hand Side    
    if(nu < lsize - 1):
      id = np.eye(2**(lsize-nu-1),2**(lsize-nu-1))
      ke_x, ke_y, ke_z = \
      np.kron(ke_x, id), np.kron(ke_y, id), np.kron(ke_z, id)
    return (ke_x, ke_y, ke_z) 
  
  def offd_corrmats(self, sitepair):
    lsize = self.lattice_size
    (mu, nu) = sitepair
    #Left Hand Side    
    if(mu == 0):
	cxy, cxz, cyz = sig_x, sig_x, sig_y 
    else:
	id = np.eye(2**mu,2**mu)
	cxy, cxz, cyz = \
	  np.kron(id, sig_x), np.kron(id, sig_x), np.kron(id, sig_y)
    #Middle Side
    dim = 1 if mu == nu else 2**(np.abs(mu-nu)-1) 
    id = np.eye(dim,dim)
    cxy, cxz, cyz = \
      np.kron(cxy, id), np.kron(cxz, id), np.kron(cyz, id)
    cxy, cxz, cyz = \
      np.kron(cxy, sig_y), np.kron(cxz, sig_z), np.kron(cyz, sig_z) 
    #Right Hand Side    
    if(nu < self.lattice_size - 1):
      id = np.eye(2**(lsize-nu-1),2**(lsize-nu-1))
      cxy, cxz, cyz = \
      np.kron(cxy, id), np.kron(cxz, id), np.kron(cyz, id)
    return (cxy, cxz, cyz) 
  
  def __init__(self, params):
    #Copy arguments from params to this class
    self.__dict__.update(params.__dict__)
    #Build KE matrix
    self.hamiltmat = self.jx * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[0] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
    self.hamiltmat += self.jy * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[1] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
    self.hamiltmat += self.jz * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[2] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
    self.hamiltmat = self.hamiltmat/self.norm
    #Build transverse field matrix
    self.hamiltmat += self.hx * np.sum(np.array([self.nummats(mu)[0] \
      for mu in xrange(self.lattice_size)]), axis=0)
    self.hamiltmat += self.hy * np.sum(np.array([self.nummats(mu)[1] \
      for mu in xrange(self.lattice_size)]), axis=0)
    self.hamiltmat += self.hz * np.sum(np.array([self.nummats(mu)[2] \
      for mu in xrange(self.lattice_size)]), axis=0)

def jac(y, t0, jacmat):
  return jacmat

def func(y, t0, jacmat):
  return np.dot(jac(y, t0, jacmat), y)

def evolve_numint(hamilt,times,initstate):
  (rows,cols) = hamilt.hamiltmat.shape
  fulljac = np.zeros((2*rows,2*cols))
  fulljac[0:rows, 0:cols] = hamilt.hamiltmat.imag
  fulljac[0:rows, cols:] = hamilt.hamiltmat.real
  fulljac[rows:, 0:cols] = -hamilt.hamiltmat.real
  fulljac[rows:, cols:] = hamilt.hamiltmat.imag
  
  psi_t = odeint(func, np.concatenate((initstate, np.zeros(rows))),\
    times, args=(fulljac,), Dfun=jac)
  return psi_t[:,0:rows] + (1j) * psi_t[:, rows:]
  
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
  lsize = h.lattice_size
  lsq = lsize * lsize
  
  #Assume that psi_0 is the eigenstate of \sum_\mu sigma^x_\mu
  initstate =  np.ones(2**lsize)/np.sqrt(2**lsize)
  dt = (t_final-t_init)/(n_steps-1.0)
  t_output = np.arange(t_init, t_final, dt)
  
  #Required observables
        
  sx = np.sum(np.array([h.nummats(mu)[0] \
    for mu in xrange(h.lattice_size)]), axis=0)
  sy = np.sum(np.array([h.nummats(mu)[1] \
    for mu in xrange(h.lattice_size)]), axis=0)
  sz = np.sum(np.array([h.nummats(mu)[2] \
    for mu in xrange(h.lattice_size)]), axis=0)
  sx, sy, sz = sx/lsize, sy/lsize, sz/lsize
  
  offset = np.eye(2**lsize, dtype=complex)
  np.fill_diagonal(offset, (1./lsize))
  
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
  
  psi_t = evolve_numint(h, t_output, initstate)
    
  sxdata = np.array([np.vdot(psi,np.dot(sx,psi)) for psi in psi_t])
  sydata = np.array([np.vdot(psi,np.dot(sy,psi)) for psi in psi_t])
  szdata = np.array([np.vdot(psi,np.dot(sz,psi)) for psi in psi_t])
  
  sxvar_data = np.array([np.vdot(psi,np.dot(sxvar,psi)) \
    for psi in psi_t])
  sxvar_data = sxvar_data - (sxdata)**2
  
  syvar_data = np.array([np.vdot(psi,np.dot(syvar,psi)) \
    for psi in psi_t])
  syvar_data = syvar_data - (sydata)**2
  
  szvar_data = np.array([np.vdot(psi,np.dot(szvar,psi)) \
    for psi in psi_t])
  szvar_data = szvar_data - (szdata)**2
  
  sxyvar_data = np.array([np.vdot(psi,np.dot(sxyvar,psi)) \
    for psi in psi_t])
  sxyvar_data = sxyvar_data - (sxdata) * (sydata)
  
  sxzvar_data = np.array([np.vdot(psi,np.dot(sxzvar,psi)) \
    for psi in psi_t])
  sxzvar_data = sxzvar_data - (sxdata) * (szdata)
  
  syzvar_data = np.array([np.vdot(psi,np.dot(syzvar,psi)) \
    for psi in psi_t])
  syzvar_data = syzvar_data - (sydata) * (szdata)
    
  data = OutData(t_output, np.abs(sxdata), np.abs(sydata), \
    np.abs(szdata), np.abs(sxvar_data), np.abs(syvar_data), \
      np.abs(szvar_data), np.abs(sxyvar_data), \
	    np.abs(sxzvar_data), np.abs(syzvar_data), params)

  print "\nDumping outputs to files ..."
  data.dump_data()
  print 'Done'

if __name__ == '__main__':
  args_in = input()
  paramdat = ParamData(args_in)
  runising_dyn(paramdat)

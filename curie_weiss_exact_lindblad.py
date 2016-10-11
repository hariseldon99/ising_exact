#!/usr/bin/env python
"""
Created on May 14 2015

@author: Analabha Roy (daneel@utexas.edu)

Usage : ./curie_weiss_exact_lindblad.py -h
"""
import argparse
import numpy as np
from scipy.sparse import dia_matrix
from pprint import pprint
from itertools import combinations
from qutip import *

#Default Parameters are entered here
t_init = 0.0 # Initial time
t_final = 10.0 # Final time
n_steps = 3000 # Number of time steps

desc = """Lindblad dissipative dynamics by exact diagonalization of 
	    1d generalized Curie-Weiss model with long range interactions. See 
          [1] for details
          Refs:
          [1] M. Foss-Feig et. al. Physical Review A 87, 042101 (2013).
          [2] P.D. Nation, J.R. Johansson, Qutip Users Guide (2011).
       """

def input():
  parser = argparse.ArgumentParser(description=desc)
  
  parser.add_argument('-l', '--lattice_size', type=np.int64,\
    help="size of the lattice", default=6)
  
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
    help="power law index for long range interactions", default=1.0)

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
    
  parser.add_argument('-gud','--gud', type=np.float64, \
    help="forward raman dissipation", default=0.0)
  parser.add_argument('-gdu','--gdu', type=np.float64, \
    help="forward raman dissipation", default=0.0)
  parser.add_argument('-gel','--gel', type=np.float64, \
    help="rayleigh dissipation", default=0.0)    
    
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
  
  def dissmats(self, mu):
      """See refs [1,2] for details on the meaning of the coefficients
      scaling the return tuple"""
      lsize = self.lattice_size
	#Left Hand Side
      if(mu == 0):
	  sm, sp, sz  = sigmam(), sigmap(), sigmaz()
      else:
        id = tensor([identity(2) for i in xrange(mu)])
        sm, sp, sz  = \
	    tensor(id,sigmam()), tensor(id,sigmap()), tensor(id,sigmaz())
	#Right Hand Side    
      if(mu < lsize - 1):
          id = tensor([identity(2) for i in xrange(lsize-mu-1)])
          sm, sp, sz = \
	      tensor(sm, id), tensor(sp, id), tensor(sz, id)    
      return (np.sqrt(self.gud) * sm, np.sqrt(self.gdu) * sp, \
                                                 np.sqrt(self.gel/4) * sz)
      
  def nummats(self, mu):
      lsize = self.lattice_size
	#Left Hand Side
      if(mu == 0):
	  num_x, num_y, num_z  = sigmax(), sigmay(), sigmaz()
      else:
        id = tensor([identity(2) for i in xrange(mu)])  
        num_x, num_y, num_z  = \
	    tensor(id,sigmax()), tensor(id,sigmay()), tensor(id,sigmaz())
      #Right Hand Side    
      if(mu < lsize - 1):
          id = tensor([identity(2) for i in xrange(lsize-mu-1)])
          num_x, num_y, num_z = \
	      tensor(num_x, id), tensor(num_y, id), tensor(num_z, id)   
      return (num_x, num_y, num_z)
  
  def kemats(self, sitepair):
    lsize = self.lattice_size
    (mu, nu) = sitepair
    #Left Hand Side    
    if(mu == 0):
        ke_x, ke_y, ke_z = sigmax(), sigmay(), sigmaz()
    else:
	id = tensor([identity(2) for i in xrange(mu)])
	ke_x, ke_y, ke_z = \
	  tensor(id, sigmax()), tensor(id, sigmay()), tensor(id, sigmaz())
    #Middle Side
    if np.abs(mu-nu) == 1:
        ke_x, ke_y, ke_z = \
            tensor(ke_x, sigmax()), tensor(ke_y, sigmay()), tensor(ke_z, sigmaz())  
    else:
        id = tensor([identity(2) for i in xrange(np.abs(mu-nu)-1)])
        ke_x, ke_y, ke_z = \
        tensor(ke_x, id), tensor(ke_y, id), tensor(ke_z, id)
        ke_x, ke_y, ke_z = \
            tensor(ke_x, sigmax()), tensor(ke_y, sigmay()), tensor(ke_z, sigmaz())    
    #Right Hand Side    
    if(nu < lsize - 1):
      id = tensor([identity(2) for i in xrange(lsize-nu-1)]) 
      ke_x, ke_y, ke_z = \
      tensor(ke_x, id), tensor(ke_y, id), tensor(ke_z, id)
    return (ke_x, ke_y, ke_z) 
  
  def offd_corrmats(self, sitepair):
    lsize = self.lattice_size
    (mu, nu) = sitepair
    #Left Hand Side    
    if(mu == 0):
	cxy, cxz, cyz = sigmax(), sigmax(), sigmay() 
    else:
      id = tensor([identity(2) for i in xrange(mu)])  
      cxy, cxz, cyz = \
	  tensor(id, sigmax()), tensor(id, sigmax()), tensor(id, sigmay())
    #Middle Side
    if np.abs(mu-nu) == 1:
        cxy, cxz, cyz = \
          tensor(cxy, sigmay()), tensor(cxz, sigmaz()), tensor(cyz, sigmaz()) 
    else:
        id = tensor([identity(2) for i in xrange(np.abs(mu-nu)-1)])
        cxy, cxz, cyz = \
          tensor(cxy, id), tensor(cxz, id), tensor(cyz, id)
        cxy, cxz, cyz = \
          tensor(cxy, sigmay()), tensor(cxz, sigmaz()), tensor(cyz, sigmaz()) 
    #Right Hand Side    
    if(nu < self.lattice_size - 1):
      id = tensor([identity(2) for i in xrange(lsize-nu-1)])
      cxy, cxz, cyz = \
      tensor(cxy, id), tensor(cxz, id), tensor(cyz, id)
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
  initstate=0.5*(identity(2)+sigmax())
  initstate=tensor([initstate for i in xrange(lsize)])
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
  
  sxvar = np.sum(np.array( [h.kemats(sitepair)[0] \
	for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
  syvar = np.sum(np.array( [h.kemats(sitepair)[1] \
    for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
  szvar = np.sum(np.array( [h.kemats(sitepair)[2] \
    for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
  sxvar, syvar, szvar = (sxvar/lsq), (syvar/lsq), (szvar/lsq)
  
  sxyvar = np.sum(np.array(\
    [ h.offd_corrmats(sitepair)[0] \
      for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
  sxzvar = np.sum(np.array(\
    [ h.offd_corrmats(sitepair)[1] \
      for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
  syzvar = np.sum(np.array(\
    [h.offd_corrmats(sitepair)[2] \
      for sitepair in combinations(xrange(h.lattice_size),2)]), axis=0)
  sxyvar, sxzvar, syzvar = (sxyvar/lsq), (sxzvar/lsq), (syzvar/lsq)

  #Lindblad jump operators
  j_sm = np.sum(np.array([h.dissmats(mu)[0] \
    for mu in xrange(h.lattice_size)]), axis=0)
  j_sp = np.sum(np.array([h.dissmats(mu)[1] \
    for mu in xrange(h.lattice_size)]), axis=0)  
  j_sz = np.sum(np.array([h.dissmats(mu)[2] \
    for mu in xrange(h.lattice_size)]), axis=0)      
  
  pbar = True if params.verbose else None
  result = mesolve(h.hamiltmat, initstate, t_output, [j_sm, j_sp, j_sz],\
          [sx, sy, sz, sxvar, syvar, szvar, sxyvar, sxzvar, syzvar], progress_bar=pbar)
  
  sxdata, sydata, szdata = result.expect[0], result.expect[1], result.expect[2] 

  sxvar_data, syvar_data, szvar_data = \
                          result.expect[3], result.expect[4], result.expect[5]
  
  sxvar_data = 2.0 * sxvar_data + (1./lsize) - (sxdata)**2 
  syvar_data = 2.0 * syvar_data + (1./lsize) - (sydata)**2
  szvar_data = 2.0 * szvar_data + (1./lsize) - (szdata)**2 
  
  sxyvar_data, sxzvar_data, syzvar_data = \
                          result.expect[6], result.expect[7], result.expect[8]

  sxyvar_data = 2.0 * sxyvar_data - (sxdata) * (sydata)
  sxzvar_data = 2.0 * sxzvar_data - (sxdata) * (szdata)
  syzvar_data = 2.0 * syzvar_data - (sydata) * (szdata)
     
  data = OutData(t_output,sxdata.real, sydata.real, \
    szdata.real, sxvar_data.real, syvar_data.real, \
      szvar_data.real, sxyvar_data.real, \
	    sxzvar_data.real, syzvar_data.real, params)
  if params.verbose:
      print "\nDumping outputs to files ..."
  data.dump_data()
  if params.verbose:
      print 'Done'

if __name__ == '__main__':
  args_in = input()
  paramdat = ParamData(args_in)
  runising_dyn(paramdat)
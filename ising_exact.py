#!/usr/bin/python

"""
Created on Tues Mar  4 2014

@author: Analabha Roy (daneel@utexas.edu)

Usage : ./isingrand_esys.py -h
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from pprint import pprint
from itertools import combinations

"""
Default Parameters are entered here
"""
#Lattice size
L = 7
t_init = 0.0 # Initial time
t_final = 60.0 # Final time
n_steps = 1000 # Number of time steps

#Power law decay of interactions
beta = 1.0

desc = """exact Diagonalization for 1d Ising model with
		long range interactions"""

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
  J = np.sum(np.diagflat(np.full(lsize-i, pow(i, -args.beta)), i) \
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
  J = np.sum(np.diagflat(np.full(lsize-i, pow(i, -args.beta)), i) \
    for i in xrange(1, mid_diag+1))+\
      np.sum(np.diagflat(np.full(lsize-i, pow(lsize-i, -args.beta)), i)\
	for i in xrange(mid_diag+1, lsize))
  J = J + J.T
  np.fill_diagonal(J, 0)
  return J

class ParamData:
  description = """Class to store parameters and time-independent
		      part of Jacobian. Set "s_order" to True 
						when doing 2nd order"""
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
      id = np.eye(2**(L-nu-1),2**(L-nu-1))
      cxy, cxz, cyz = \
      np.kron(cxy, id), np.kron(cxz, id), np.kron(cyz, id)
    return (cxy, cxz, cyz) 
  
  def __init__(self, params):
    #Copy arguments from params to this class
    self.__dict__.update(params.__dict__)
    #Build KE matrix
    H = self.jx * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[0] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
    H += self.jy * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[1] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
    H += self.jz * np.sum(np.array(\
      [self.jmat[sitepair] * self.kemats(sitepair)[2] \
	for sitepair in combinations(xrange(self.lattice_size),2)]), axis=0)
    H = H/self.norm
    #Build transverse field matrix
    H += self.hx * np.sum(np.array([self.nummats(mu)[0] \
      for mu in xrange(self.lattice_size)]), axis=0)
    H += self.hy * np.sum(np.array([self.nummats(mu)[1] \
      for mu in xrange(self.lattice_size)]), axis=0)
    H += self.hz * np.sum(np.array([self.nummats(mu)[2] \
      for mu in xrange(self.lattice_size)]), axis=0)
    try:
	evals, U = np.linalg.eigh(-H)
	idx = evals.argsort()
	evals = evals[idx]
	U = U[:,idx]
	self.esys = (evals, U)
    except np.linalg.linalg.LinAlgError:
	self.esys = None

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
	  
#Pauli matrices
sig_x,sig_y,sig_z = np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),\
np.array([[1,0],[0,-1]])                    

def runising_dyn(params):
  if params.verbose:
    print "Executing diagonalization with parameters:"
    pprint(vars(params), depth=1)
  else:
    print "Starting run ..."
  hamilt = Hamiltonian(params)  
  
  if hamilt.esys is not None:
    
    lsize = hamilt.lattice_size
    lsq = lsize * lsize
    
    sx = np.sum(np.array([hamilt.nummats(mu)[0] \
      for mu in xrange(hamilt.lattice_size)]), axis=0)
    sy = np.sum(np.array([hamilt.nummats(mu)[1] \
      for mu in xrange(hamilt.lattice_size)]), axis=0)
    sz = np.sum(np.array([hamilt.nummats(mu)[2] \
      for mu in xrange(hamilt.lattice_size)]), axis=0)
    sx, sy, sz = sx/lsize, sy/lsize, sz/lsize
    
    sxvar = np.sum(np.array(\
      [hamilt.jmat[sitepair] * hamilt.kemats(sitepair)[0] \
	for sitepair in combinations(\
	  xrange(hamilt.lattice_size),2)]), axis=0)
    syvar = np.sum(np.array(\
      [hamilt.jmat[sitepair] * hamilt.kemats(sitepair)[1] \
	for sitepair in combinations(\
	  xrange(hamilt.lattice_size),2)]), axis=0)
    szvar = np.sum(np.array(\
      [hamilt.jmat[sitepair] * hamilt.kemats(sitepair)[2] \
	for sitepair in combinations(\
	  xrange(hamilt.lattice_size),2)]), axis=0)
    sxvar, syvar, szvar = sxvar/lsq, syvar/lsq, szvar/lsq
    
    sxyvar = np.sum(np.array(\
      [hamilt.jmat[sitepair] * hamilt.offd_corrmats(sitepair)[0] \
	for sitepair in combinations(\
	  xrange(hamilt.lattice_size),2)]), axis=0)
    sxzvar = np.sum(np.array(\
      [hamilt.jmat[sitepair] * hamilt.offd_corrmats(sitepair)[1] \
	for sitepair in combinations(\
	  xrange(hamilt.lattice_size),2)]), axis=0)
    syzvar = np.sum(np.array(\
      [hamilt.jmat[sitepair] * hamilt.offd_corrmats(sitepair)[2] \
	for sitepair in combinations(\
	  xrange(hamilt.lattice_size),2)]), axis=0)
    sxyvar, sxzvar, syzvar = sxyvar/lsq, sxzvar/lsq, syzvar/lsq
    
    #mag_(x,y,z) are the magnetization operators in the diagonal basis
    mag_x = np.transpose(hamilt.esys[1]).dot(sx.dot(hamilt.esys[1]))
    mag_y = np.transpose(hamilt.esys[1]).dot(sy.dot(hamilt.esys[1]))
    mag_z = np.transpose(hamilt.esys[1]).dot(sz.dot(hamilt.esys[1]))
    
    #corr_ab are the ab correlation operators in the diagonal basis
    corr_xx = np.transpose(hamilt.esys[1]).dot(sxvar.dot(hamilt.esys[1]))
    corr_yy = np.transpose(hamilt.esys[1]).dot(syvar.dot(hamilt.esys[1]))
    corr_zz = np.transpose(hamilt.esys[1]).dot(szvar.dot(hamilt.esys[1]))
    corr_xy = np.transpose(hamilt.esys[1]).dot(sxyvar.dot(hamilt.esys[1]))
    corr_xz = np.transpose(hamilt.esys[1]).dot(sxzvar.dot(hamilt.esys[1]))
    corr_yz = np.transpose(hamilt.esys[1]).dot(syzvar.dot(hamilt.esys[1]))
    
    #Now, the expectation value of any observable A is just
    # U^\dagger(t) A U(t). In the diagonal basis, U(t) is just
    #diag(e^{i e_n t}) where e_n are the eigenvals!
    # So U(t) = np.diag(np.exp(1j*hamilt.esys[0]*t))
    #Assume that psi_0 is all spins up ie (1 0 0 0 ... 0)
    initstate = np.zeros(2**hamilt.lattice_size)
    initstate[0] = 1.0
    dt = (t_final-t_init)/(n_steps-1.0)
    t_output = np.arange(t_init, t_final, dt)
    
    sxdata = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      mag_x.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    sydata = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      mag_y.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    szdata = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      mag_z.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    sxvar_data = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      corr_xx.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    syvar_data = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      corr_yy.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    szvar_data = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      corr_zz.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    sxyvar_data = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      corr_xy.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    sxzvar_data = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      corr_xz.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    syzvar_data = np.array([np.vdot(initstate, np.diag(\
      np.exp(1j*hamilt.esys[0]*t)).T.conjugate().dot(\
      corr_yz.dot(np.diag(np.exp(1j*hamilt.esys[0]*t)))).dot(initstate)) \
	for t in t_output])
    
    data = OutData(t_output, sxdata, sydata, szdata, sxvar_data,\
      syvar_data, szvar_data, sxyvar_data, sxzvar_data, syzvar_data, params)
    
    if params.verbose:
      print "-----------------------------------"
      #Plot the eigenvalues and eigenvectors and magnetization components
      plt.plot(hamilt.esys[0].real/L)
      plt.title('Eigenvalues - Sorted')
      plt.matshow(np.abs(hamilt.esys[1])**2,interpolation='nearest',\
      cmap=cm.coolwarm)
      plt.title('Eigenvectors - Sorted' ) 
      plt.colorbar()
      fig, ax = plt.subplots()
      diag_cc = np.abs(hamilt.esys[1][0,:])**2
      plt.bar(np.delete(np.arange(2**L+1),0),diag_cc,edgecolor='blue')                
      plt.xlim(limits)
      plt.text(L,-np.max(diag_cc)/10.0,'Transverse field = %lf'%tfield,\
      horizontalalignment='center')
      #plt.title('Transverse field = %lf'%tfield )
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
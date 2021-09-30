"""
mainExactDiag.py
---------------------------------------------------------------------
Script file for initializing exact diagonalization using the 'eigsh' routine
for a 1D quantum system.

by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 06/2020
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer

from doApplyHam import doApplyHam

# Simulation parameters
model = 'XX'  # select 'XX' model of 'ising' model
Nsites = 18  # number of lattice sites
usePBC = True  # use periodic or open boundaries
numval = 1  # number of eigenstates to compute

# Define Hamiltonian (quantum XX model)
d = 2  # local dimension
sX = np.array([[0, 1.0], [1.0, 0]])
sY = np.array([[0, -1.0j], [1.0j, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])
if model == 'XX':
  hloc = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
  EnExact = -4 / np.sin(np.pi / Nsites)  # Note: only for PBC
elif model == 'ising':
  hloc = (-np.kron(sX, sX) + 0.5 * np.kron(sZ, sI) + 0.5 * np.kron(sI, sZ)
          ).reshape(2, 2, 2, 2)
  EnExact = -2 / np.sin(np.pi / (2 * Nsites))  # Note: only for PBC


# cast the Hamiltonian 'H' as a linear operator
def doApplyHamClosed(psiIn):
  return doApplyHam(psiIn, hloc, Nsites, usePBC)


H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

# do the exact diag
start_time = timer()
Energy, psi = eigsh(H, k=numval, which='SA')
diag_time = timer() - start_time

# check with exact energy
EnErr = Energy[0] - EnExact  # should equal to zero

print('NumSites: %d, Time: %1.2f, Energy: %e, EnErr: %e' %
      (Nsites, diag_time, Energy[0], EnErr))


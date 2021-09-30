# doApplyHam.py
# ---------------------------------------------------------------------
# Routine used in the implementation of exact diagonalization.
#
# by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 6/2019

import numpy as np


def doApplyHam(psiIn: np.ndarray,
               hloc: np.ndarray,
               N: int,
               usePBC: bool):
  """
  Applies local Hamiltonian, given as sum of nearest neighbor terms, to
  an input quantum state.
  Args:
    psiIn: vector of length d**N describing the quantum state.
    hloc: array of ndim=4 describing the nearest neighbor coupling.
    N: the number of lattice sites.
    usePBC: sets whether to include periodic boundary term.
  Returns:
    np.ndarray: state psi after application of the Hamiltonian.
  """
  d = hloc.shape[0]
  psiOut = np.zeros(psiIn.size)
  for k in range(N - 1):
    # apply local Hamiltonian terms to sites [k,k+1]
    psiOut += np.tensordot(hloc.reshape(d**2, d**2),
                           psiIn.reshape(d**k, d**2, d**(N - 2 - k)),
                           axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)

  if usePBC:
    # apply periodic term
    psiOut += np.tensordot(hloc.reshape(d, d, d, d),
                           psiIn.reshape(d, d**(N - 2), d),
                           axes=[[2, 3], [2, 0]]
                           ).transpose(1, 2, 0).reshape(d**N)

  return psiOut


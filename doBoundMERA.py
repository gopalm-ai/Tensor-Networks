# -*- coding: utf-8 -*-
# doBoundMERA.py
import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
from ncon import ncon


def doBoundMERA(gR,hamR,rhoR,qC,chiB,numlevels, numiter=3000, dispon=True, sciter=4):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 25/1/2019
------------------------
Variational energy minimization of the tensors 'qC' at the edge of a \
boundary MERA. Inputs 'gR' and 'hamR' define the boundary Hamiltonian (on \
the log-scale lattice), while 'rhoR' is the one-site density matrix for \
the boundary MPS. The dimension of the boundary MPS is set by 'chiB' \
while 'numlevels' sets the number of boundary transitional layers.

Optional arguments:
`numiter::Int=3000`: number of variatonal iterations
`dispon::Bool=True`: specify wether or not to display convergence data
`sciter::Int=4`: iterations of power method to find scale-invariant density matrix
"""
    ##### Expand tensors to new dimensions if required
    for k in range(numlevels - len(qC)):
        qC.append(qC[-1])
        rhoR.append(rhoR[-1])
    
    for k in range(numlevels-1):
        chiL = qC[k].shape[0]
        chiR = qC[k].shape[1]
        chitemp = min(chiB,chiL*chiR)
        if qC[k].shape[2] < chitemp:
            qC[k] = TensorExpand(qC[k],[chiL,chiR,chitemp])
            qC[k+1] = TensorExpand(qC[k+1],[chitemp,qC[k+1].shape[1],qC[k+1].shape[2]])
            
    qC[numlevels-1] = qC[numlevels-2]
    if rhoR[numlevels].shape[0] != qC[numlevels-1].shape[2]:
        rhoR[numlevels] = TensorExpand(rhoR[numlevels],[qC[numlevels-1].shape[2],qC[numlevels-1].shape[2]])

    ##### Ensure Hamiltonian is negative defined
    chiL = gR[0].shape[0]; chiR = gR[0].shape[1]
    bias = max(LA.eigvalsh(gR[0].reshape(chiL*chiR,chiL*chiR)))
    gR[0] = gR[0] - bias * np.eye(chiL*chiR).reshape(chiL,chiR,chiL,chiR)
    
    for k in range(1,numlevels+1):
        chiL = hamR[k].shape[0]; chiR = hamR[k].shape[1]
        bias = max(LA.eigvalsh(hamR[k].reshape(chiL*chiR,chiL*chiR)))
        hamR[k] = hamR[k] - bias*np.eye(chiL*chiR).reshape(chiL,chiR,chiL,chiR)

    Energy = 0
    for k in range(numiter):
        ##### Find scale-invariant density matrix (via power method)
        for p in range(sciter):
            rhoR[numlevels] = ncon([rhoR[numlevels],qC[numlevels-1],np.conj(qC[numlevels-1])],[[1,2],[-2,3,1],[-1,3,2]])
        
        rhoR[numlevels] = 0.5*(rhoR[numlevels]+(np.conj(rhoR[numlevels]).T))
        rhoR[numlevels] = rhoR[numlevels]/np.trace(rhoR[numlevels])

        ##### Descend density matrix through all layers
        for p in range(numlevels-1,-1,-1):
            rhoR[p] = ncon([rhoR[p+1],qC[p],np.conj(qC[p])],[[1,2],[-2,3,1],[-1,3,2]])
        
        ##### Optimise over all layers
        for p in range(numlevels-1):
            if k > 9:
                qenv1 = ncon([gR[p],np.conj(qC[p]),qC[p+1],np.conj(qC[p+1]),rhoR[p+2]],
                              [[1,2,-1,-2],[1,2,6],[-3,4,5],[6,4,3],[3,5]]);
                qenv2 = ncon([hamR[p+1],np.conj(qC[p]),qC[p+1],np.conj(qC[p+1]),rhoR[p+2]],
                              [[5,3,-2,4],[-1,5,6],[-3,4,2],[6,3,1],[1,2]]);
                qC[p] = TensorUpdateSVD(qenv1+qenv2,2)

            chitemp = hamR[p+1].shape[1]
            gRtemp = ncon([gR[p],qC[p],np.conj(qC[p]),np.eye(chitemp)], [[1,2,3,4],[3,4,-1],
              [1,2,-3],[-2,-4]]).reshape(qC[p].shape[2],chitemp,qC[p].shape[2],chitemp) 
            gR[p+1] = gRtemp + ncon([hamR[p+1],qC[p],np.conj(qC[p])],[[1,-2,2,-4],[3,2,-3],[3,1,-1]])
            
        qC[numlevels-1] = qC[numlevels-2]
        
        ##### Compute energy and display
        if dispon:
            if np.mod(k,10) == 0:
                boundSuper = ncon([qC[-1],np.conj(qC[-1])],[[-2,1,-4],
                                   [-1,1,-3]]).reshape((qC[-1].shape[2])**2,(qC[-1].shape[2])**2)
                dtemp = eigs(boundSuper,k=6,which='LM')
                scDims = -np.log2(abs(dtemp[0]))
                Energy = ncon([gR[numlevels-1],qC[numlevels-1],np.conj(qC[numlevels-1]),rhoR[numlevels]],
                               [[1,2,3,4],[3,4,5],[1,2,6],[6,5]])

                print('Iteration: %d of %d, Bond dim: %d, Energy: %f, ScDim: %e' % (k,numiter,chiB,Energy,scDims[1]))
                
    return Energy, qC, rhoR

"""
TensorExpand: expand tensor dimension by padding with zeros
"""
def TensorExpand(A,chivec):
    
    if [*A.shape] == chivec:
        return A
    else:
        for k in range(len(chivec)):
            if A.shape[k] != chivec[k]:
                indloc = list(range(-1,-len(chivec)-1,-1))
                indloc[k] = 1
                A = ncon([A,np.eye(A.shape[k],chivec[k])],[indloc,[1,-k-1]])
                
        return A
    

"""
TensorUpdateSVD: update an isometry or unitary tensor using its \
(linearized) environment
"""
def TensorUpdateSVD(wIn,leftnum):

    wSh = wIn.shape
    ut,st,vht = LA.svd(wIn.reshape(np.prod(wSh[0:leftnum:1]),
                                   np.prod(wSh[leftnum:len(wSh):1])),full_matrices=False)
    return -(ut @ vht).reshape(wSh)

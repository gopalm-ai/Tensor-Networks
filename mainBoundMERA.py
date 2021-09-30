# -*- coding: utf-8 -*-
""" 
mainBoundMERA.py
---------------------------------------------------------------------
Script file for initializing a boundary MERA calculation. Requires \
isometries 'wC' and disentanglers 'uC' from a scale-invariant modified \
binary MERA (with spatial reflection symmetry). Works with the default \
output of the MERA code at "www.tensors.net". This example code optimizes \
for the critical Ising model with free boundary conditions, then computes \
the boundary scaling dimensions, boundary energy and boundary \
magenetization.

by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 25/1/2019
"""

#### Preamble
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

from ncon import ncon
from doBoundMERA import doBoundMERA 

##### Example 1: critical Ising model with free boundary
###########################################################

chiB = 8 # boundary bond dimension
boundlayers = 6 # number of transitional layers on the boundary

OPTS_dispon = True # print convergence data
OPTS_numiter = 3000 # number of iterations
OPTS_sciter = 4 # iterations of power method to find scale-invariant density matrix

##### Load data from bulk MERA calculation if necessary
wC,uC,vC,rhoAB,rhoBA = np.load('IsingData.npy')

##### Define Hamiltonian (critical Ising)
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0,-1]])
htemp1 = -np.kron(sX,sX)
htemp2 = -np.kron(sX,sX) - np.kron(sZ,np.eye(2)) - np.kron(np.eye(2),sZ)
hamAB = [0]
hamAB[0] = (np.kron(np.eye(4),htemp2) + np.kron(np.eye(2),np.kron(htemp1,np.eye(2))) + 
     np.kron(htemp2,np.eye(4))).reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6).reshape(4,4,4,4)
hamBA = [0]
hamBA[0] = np.kron(np.eye(2),np.kron(htemp1,np.eye(2))).reshape(2,2,2,2,2,2,2,2).transpose(1,0,2,3,5,4,6,7).reshape(4,4,4,4)

##### Use MERA tensors to coarse-grain Hamiltonian
indList1 = [[6,4,1,2],[1,3,-3],[6,7,-1],[2,5,3,9],[4,5,7,10],[8,9,-4],[8,10,-2]]
indList2 = [[3,4,1,2],[5,6,-3],[5,7,-1],[1,2,6,9],[3,4,7,10],[8,9,-4],[8,10,-2]]
indList3 = [[5,7,2,1],[8,9,-3],[8,10,-1],[4,2,9,3],[4,5,10,6],[1,3,-4],[7,6,-2]]
indList4 = [[3,6,2,5],[2,1,-3],[3,1,-1],[5,4,-4],[6,4,-2]]
maxlayers = 20
for k in range(maxlayers):
    if (k+1) > len(wC):
        wC.append(wC[-1])
        uC.append(uC[-1])
    
    hamBAout = ncon([hamAB[k],wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList1)
    hamBAout = hamBAout + hamBAout.transpose(1,0,3,2)
    hamBAout = hamBAout + ncon([hamBA[k],wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList2)
    hamBA.append(hamBAout)
    hamABout = ncon([hamBA[k],wC[k],np.conj(wC[k]),wC[k],np.conj(wC[k])],indList4)
    hamAB.append(hamABout)

##### Define log-scale Hamiltonian (with free BC)
gR = [0 for x in range(boundlayers)]
hamR = [0 for x in range(boundlayers+1)]
gR[0] = hamAB[0]
for k in range(boundlayers):
    hamR[k+1] = ncon([hamBA[k],wC[k],np.conj(wC[k])],[[-1,1,-3,2],[2,3,-4],[1,3,-2]])

##### Initialize tensors
qC = [0 for x in range(boundlayers)]
qC[0] = np.random.rand(hamAB[0].shape[0],hamAB[0].shape[1],min(chiB,hamAB[0].shape[0]*hamAB[0].shape[1]))
for k in range(1,boundlayers):
    qC[k] = np.random.rand(qC[k-1].shape[2],hamR[k].shape[1],min(chiB,qC[k-1].shape[2]*hamR[k].shape[1]))
    
rhoR = [0 for x in range(boundlayers+1)]
rhoR[boundlayers] = np.eye(qC[boundlayers-1].shape[2])

##### Perform variational optimization for boundary ####
Energy, qC, rhoR = doBoundMERA(gR,hamR,rhoR,qC,chiB,boundlayers,
                               dispon = OPTS_dispon,numiter = OPTS_numiter,sciter = OPTS_sciter)
########################################################

"""
evalBoundMERA: evaluates outputs from boundary MERA algorithm to compute \
several quantities of interest (scaling dimensions, boundary energy \
correction, boundary magnetization profile)
"""
def evalBoundMERA(rhoR, qC, wC, uC, hamBA, hamAB, rhoAB, rhoBA, boundlayers):

    ##### Evaluate boundary scaling dimensions
    boundlayers = len(qC);
    boundSuper = ncon([qC[-1],np.conj(qC[-1])],[[-2,1,-4],[-1,1,-3]]).reshape((qC[-1].shape[2])**2,(qC[-1].shape[2])**2)
    
    dtemp = eigs(boundSuper,k=6,which='LM')
    scDims = -np.log2(abs(dtemp[0]))
                
    ##### Evaluate boundary contribution to the energy
    rhoBAnew = [0 for x in range(boundlayers-1)]
    for k in range(boundlayers-2,-1,-1):
        rhotemp = ncon([qC[k],np.conj(qC[k]),qC[k+1],np.conj(qC[k+1]),rhoR[k+2]],
                        [[3,-3,4],[3,-1,5],[4,-4,2],[5,-2,1],[1,2]])
        rhoBAnew[k] = ncon([rhotemp,wC[k],np.conj(wC[k])],[[-1,1,-3,2],[-2,3,1],[-4,3,2]])
    
    rhoABnew = [0 for x in range(boundlayers-1)]
    rhoABnew[0] = ncon([qC[0],np.conj(qC[0]),rhoR[1]],[[-3,-4,2],[-1,-2,1],[1,2]])
    
    Ebound = np.zeros(len(hamBA)+1)
    Ebound[0] = ncon([rhoABnew[0],hamAB[0]],[[1,2,3,4],[1,2,3,4]])
    for k in range(len(hamBA)):
        if len(rhoBAnew) < (k+1):
            rhoBAnew.append(rhoBAnew[-1])
        
        Ebound[k+1] = ncon([rhoBAnew[k],hamBA[k]],[[1,2,3,4],[1,2,3,4]])
    
    
    Ebulk = np.zeros(len(hamBA)+1)
    Ebulk[0] = ncon([rhoAB[0],hamAB[0]],[[1,2,3,4],[1,2,3,4]]) + 0.5*ncon([hamBA[0],rhoBA[0]],[[1,2,3,4],[1,2,3,4]])
    for k in range(len(hamBA)):
        if len(rhoBA) < (k+1):
            rhoBA.append(rhoBA[k-1])
        
        Ebulk[k+1] = ncon([rhoBA[k],hamBA[k]],[[1,2,3,4],[1,2,3,4]])
    
    EnBoundCorr = sum(Ebound-Ebulk)
    
    ##### Compute local reduced density matrices for boundary system
    indList1 = [[9,3,4,2],[-3,5,4],[-1,10,9],[-4,7,5,6],[-2,7,10,8],[1,6,2],[1,8,3]]
    indList2 = [[3,6,2,5],[1,7,2],[1,9,3],[-3,-4,7,8],[-1,-2,9,10],[4,8,5],[4,10,6]]
    indList3 = [[3,9,2,4],[1,5,2],[1,8,3],[7,-3,5,6],[7,-1,8,10],[-4,6,4],[-2,10,9]]
    indList4 = [[3,6,2,5],[-3,1,2],[-1,1,3],[-4,4,5],[-2,4,6]]
    
    rhoABbott = [0 for x in range(11)]
    rhoABbott[0] = rhoABnew[0].reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    
    k = 0
    rhoABtemp = ncon([rhoBAnew[k+1],wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList1)
    rhoABbott[1] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    rhoABtemp = ncon([rhoBAnew[k+1],wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList3)
    rhoABbott[2] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    
    k = 1
    rhoBAtemp = ncon([rhoBAnew[k+1],wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList2)
    k = 0
    rhoABtemp = ncon([rhoBAtemp,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList1)
    rhoABbott[3] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    rhoABtemp = ncon([rhoBAtemp,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList3)
    rhoABbott[4] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    
    k = 2
    rhoABtempL = ncon([rhoBAnew[k+1],wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList1)
    rhoBAtempM = ncon([rhoBAnew[k+1],wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList2)
    rhoABtempR = ncon([rhoBAnew[k+1],wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList3)
    k = 1
    rhoBAtempL = ncon([rhoABtempL,wC[k],np.conj(wC[k]),wC[k],np.conj(wC[k])],indList4)
    rhoBAtempM = ncon([rhoBAtempM,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList2)
    rhoBAtempR = ncon([rhoABtempR,wC[k],np.conj(wC[k]),wC[k],np.conj(wC[k])],indList4)
    k = 0
    rhoABtemp = ncon([rhoBAtempL,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList1)
    rhoABbott[5] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    rhoABtemp = ncon([rhoBAtempL,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList3)
    rhoABbott[6] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    rhoABtemp = ncon([rhoBAtempM,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList1)
    rhoABbott[7] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    rhoABtemp = ncon([rhoBAtempM,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList3)
    rhoABbott[8] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    rhoABtemp = ncon([rhoBAtempR,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList1)
    rhoABbott[9] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    rhoABtemp = ncon([rhoBAtempR,wC[k],np.conj(wC[k]),uC[k],np.conj(uC[k]),wC[k],np.conj(wC[k])],indList3)
    rhoABbott[10] = rhoABtemp.reshape(2,2,2,2,2,2,2,2).transpose(0,1,3,2,4,5,7,6)
    
    ##### Evaluate boundary magnetization
    magz = np.zeros(44)
    sZ = np.array([[1, 0], [0,-1]])
    for p in range(11):
        magz[4*(p)] = ncon([rhoABbott[p],sZ],[[4,1,2,3,5,1,2,3],[4,5]])
        magz[1+4*(p)] = ncon([rhoABbott[p],sZ],[[1,4,2,3,1,5,2,3],[4,5]])
        magz[2+4*(p)] = ncon([rhoABbott[p],sZ],[[1,2,4,3,1,2,5,3],[4,5]])
        magz[3+4*(p)] = ncon([rhoABbott[p],sZ],[[1,2,3,4,1,2,3,5],[4,5]])

    return scDims, EnBoundCorr, magz


#------------------------------------------------
# Evaluate obserables of interest
scDims, EnBoundCorr, magz = evalBoundMERA(rhoR, qC, wC, uC, hamBA, hamAB, rhoAB, rhoBA, boundlayers)

BoundCorrExact = (1/2 - 1/np.pi); # exact boundary energy correction
print('----------------- Final Results -----------------')
print('BoundEn_ex: %f, BoundEn_MERA: %f, EnErr: %e' % (BoundCorrExact,EnBoundCorr,abs(BoundCorrExact-EnBoundCorr)))

##### Scaling dimensions
scDimsExact = [0,0.5,1.5,2,2.5,3]
plt.figure(1)
plt.plot(range(6), scDimsExact, 'b', label="exact")
plt.plot(range(6), scDims, 'rx', label="MERA")
plt.legend()
plt.title('critical Ising model')
plt.xlabel('k')
plt.ylabel('Boundary Scaling Dims: Delta_k')
plt.show()

##### Plot boundary magnetization
magzexact = (2/np.pi)*(1 + 1/(4*np.array(range(44))+3))
magerr = magz - magzexact
plt.figure(2)
plt.plot(range(len(magzexact)), magzexact, 'b', label="exact")
plt.plot(range(len(magz)), magz, 'rx', label="MERA")
plt.legend()
plt.title('Boundary Magnetization from MERA')
plt.xlabel('Distance from boundary')
plt.ylabel(' Magnetization')
plt.show()


# -*- coding: utf-8 -*-
""" mainPEPS.py
---------------------------------------------------------------------
Script file for initializing the Hamiltonian and PEPS tensors (2D square \
lattice with 2-site unit cell) before passing to the TEBD routine for PEPS.

    by Glen Evenbly (c) for www.tensors.net, (v1.1) - last modified 29/1/2019
"""

#### Preamble
import numpy as np
from numpy import linalg as LA
from doPEPS_TEBD import doPEPS_TEBD 

whichExample = 1;

if whichExample == 1:
    ######################################
    ##### Example 1: Heisenberg model

    ##### Set bond dimensions and options
    chiD = 4 # set PEPS virtual dimension
    chiM = 8 # PEPS boundary dimension
    taustep = 0.1 # time-step

    OPTS_updateon = True # perform PEPS tensor updates
    OPTS_dispon = True # display convergence data
    OPTS_breps = 1 # number of boundary update steps between each timestep
    OPTS_numiter = 40 # total number of TEBD time-steps
    OPTS_enexact = -0.6694421 # specify exact ground energy if known

    ##### define Hamiltonian
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0,-1]])
    sI = np.eye(2)
    hloc = 0.25*np.real(np.kron(sX,sX) + np.kron(sY,sY) + np.kron(sZ,sZ))

    ##### initialize tensors
    A = 1e-2*np.random.rand(chiD,chiD,chiD,chiD,2)
    A[0,0,0,0,0] = 1
    B = 1e-2*np.random.rand(chiD,chiD,chiD,chiD,2) 
    B[0,0,0,0,1] = 1

elif whichExample == 2:
    ######################################
    ##### Example 2: Ising model

    ##### Set bond dimensions and options
    chiD = 4 # set PEPS virtual dimension
    chiM = 8 # PEPS boundary dimension
    taustep = 0.1 # time-step

    OPTS_updateon = True
    OPTS_dispon = True
    OPTS_breps = 1
    OPTS_numiter = 40
    OPTS_enexact = -3.28471

    ##### define Hamiltonian (Ising model near criticality)
    hmag = 3.1
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0,-1]])
    sI = np.eye(2)
    hloc = np.real(-np.kron(sX,sX) -hmag*0.25*(np.kron(sI,sZ)+np.kron(sZ,sI)))

    ##### initialize tensors
    A = 1e-2*np.random.rand(chiD,chiD,chiD,chiD,2)
    A[0,0,0,0,0] = 1
    B = 1e-2*np.random.rand(chiD,chiD,chiD,chiD,2) 
    B[0,0,0,0,0] = 1

##### do TEBD iterations
A,B,avEn,Tbnd = doPEPS_TEBD(A,B,chiM,hloc,taustep, numiter = OPTS_numiter,
    breps = OPTS_breps, updateon = OPTS_updateon, dispon = OPTS_dispon, enexact = OPTS_enexact)

# increase boundary dimension and reduce timestep
chiM = 16
taustep = 0.01
OPTS_numiter = 100
A,B,avEn,Tbnd = doPEPS_TEBD(A,B,chiM,hloc,taustep,Tbnd=Tbnd,numiter=OPTS_numiter,
    breps=OPTS_breps, updateon=OPTS_updateon, dispon=OPTS_dispon, enexact=OPTS_enexact)

# increase boundary dimension
chiM = 24
OPTS_numiter = 10
A,B,avEn,Tbnd = doPEPS_TEBD(A,B,chiM,hloc,taustep, Tbnd=Tbnd,numiter=OPTS_numiter,
    breps=OPTS_breps, updateon=OPTS_updateon, dispon=OPTS_dispon, enexact=OPTS_enexact)


import sys 
import argparse
import tools
import time
import numpy as np

import mpi4py
from mpi4py import MPI

import HillClimbing as HC

#MPI information extraction
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me  = comm.Get_rank()

#Process 0 prints a "hello msg"
comm.barrier()
if Me == 0:
  print("PE: ", Me, "/",NbP,": all processes started")

#Each process runs a local search method from a random starting point
S0 = HC.generateS0()
eb, Sb,iter = HC.HillClimbing(S0, 50, [0,1,2], "flops")

#Process 0 (root) gathers results (Eb, Sb), Starting points (S0) and iter nb (Iter)
# - Allocate real numpy arrays only on 0 (root) process
if Me == 0:        
  nd = HC.GetNbDim()
  EbTab = np.zeros(NbP*1,dtype=np.float64)
  SbTab = np.zeros(NbP*nd,dtype=int)
  S0Tab = np.zeros(NbP*nd,dtype=int)
  IterTab = np.zeros(NbP*1,dtype=int)
else :
  EbTab   = None     
  SbTab   = None
  S0Tab   = None
  IterTab = None

# - Gather all Eb into EbTAB, all Sb into SbTab, all S0 into S0Tab      TO DO
#
#EbTab = Eb           # Replace these lines with real gather op
#SbTab = Sb
#S0Tab = S0
#IterTab = Iter

Eb = np.array([eb],dtype=np.float64)
comm.Gather(Eb,EbTab,root=0)

comm.Gather(Sb,SbTab,root=0)
comm.Gather(S0,S0Tab,root=0)

Iter = np.array([iter],dtype=int)
comm.Gather(Iter,IterTab,root=0)


#Print results
if Me == 0:
  nd = HC.GetNbDim()
  tools.printResults(EbTab,SbTab,S0Tab,IterTab,nd,Me,NbP)

#Process 0 prints a "good bye msg"
comm.barrier()
time.sleep(1)
if Me == 0:
  print("PE: ", Me, "/",NbP," bye!")
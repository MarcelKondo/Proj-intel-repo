import simulated_annealing as SA
import random
import numpy as np
import time
from mpi4py import MPI
import general_config as GC
from server_content.automated_compiling_tabu import define_copiler_settings


#Seed for S0 generation
# seed = 15

#MPI information extraction
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me  = comm.Get_rank()

temp_la_dict={
  0: [80, 0.2],
  1: [80, 0.4],
  2: [80, 0.6],
  3: [80, 0.8]
}

define_copiler_settings(opLevel=3, simdType="avx512", version="dev13")
GC.define_neighbourhood("basic")

#Process 0 prints a "hello msg"
comm.barrier()
if Me == 0:
  print("PE: ", Me, "/",NbP,": all processes started")

T0 = temp_la_dict[Me][0]
la = temp_la_dict[Me][1]
IterMax = 20

#Each process runs a local search method from a random starting point
S0 = {
    'n1' : 512,
    'n2' : 512,
    'n3' : 512,
    'nb_threads' : 8,
    'nb_it' : 10,
    'tblock1' : 32,
    'tblock2' : 32,
    'tblock3' : 32
    #'simdType' : "avx512"
}
eb, Sb,iter = SA.SimulatedAnnealing(S0, IterMax, T0, la)

#Process 0 (root) gathers results (Eb, Sb), Starting points (S0) and iter nb (Iter)
# - Allocate real numpy arrays only on 0 (root) process
if Me == 0:        
  nd = SA.GetNbDim()
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

comm.Gather(np.array([x for x in Sb.values()]),SbTab,root=0)
comm.Gather(np.array([x for x in S0.values()]),S0Tab,root=0)

Iter = np.array([iter],dtype=int)
comm.Gather(Iter,IterTab,root=0)


#Print results
if Me == 0:
    nd = SA.GetNbDim()
    #tools.printResults(EbTab,SbTab,S0Tab,IterTab,nd,Me,NbP)
    EbTab.resize(NbP)
    SbTab.resize(NbP, nd)
    S0Tab.resize(NbP, nd)
    IterTab.resize(NbP, nd)
    print("Energies")
    print(EbTab)
    print("Optimal parameters")
    print(SbTab)
    print("Initial parameters")
    print(S0Tab)
    print("Iterations")
    print(IterTab)
#Process 0 prints a "good bye msg"
comm.barrier()
time.sleep(1)
if Me == 0:
    print("PE: ", Me, "/",NbP," bye!")
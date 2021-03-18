import os
import sys
import time
import numpy as np
import math
import random as rd

import mpi4py
from mpi4py import MPI

import HillClimbing as HC
import general_config as GC
import deploy_greedy_v3 as GR

from server_content.automated_compiling_tabu import define_copiler_settings

define_copiler_settings(opLevel=3, simdType="avx512")
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

#Process 0 prints a "hello msg"
comm.barrier()
if Me == 0:
    print("PE: ", Me, "/",NbP,": all processes started")


param_list = ['n1','n1','n1'] #parse CLI

S0 = {
    'n1' : 256,
    'n2' : 256,
    'n3' : 256,
    'nb_threads' : 4,
    'nb_it' : 50,
    'tblock1' : 32,
    'tblock2' : 32,
    'tblock3' : 32,
}

#Define the used parameters
GC.define_usedParameters(param_list)

#Process 0 (root) gathers results (Eb, Sb), Starting points (S0) and iter nb (Iter)
# - Allocate real numpy arrays only on 0 (root) process
if Me == 0:        
  nd = GC.GetNbDim()
  EbTab = np.zeros(NbP*1,dtype=np.float64)
  SbTab = np.zeros(NbP*nd,dtype=int)
  S0Tab = np.zeros(NbP*nd,dtype=int)
  IterTab = np.zeros(NbP*1,dtype=int)
else :
  EbTab   = None     
  SbTab   = None
  S0Tab   = None
  IterTab = None

#HillClimbing
if Me == 0:
    HC_eb, HC_sb, HC_iter = HC.HillClimbing(S0, S0['nb_it'], "flops")

#HC_eb = np.array([HC_eb],dtype=np.float64)
#comm.Gather(HC_eb,EbTab,root=0)

#HC_sb_a = np.fromiter(HC_sb.values(), dtype = int)
#comm.Gather(HC_sb_a,SbTab,root=0)

#S0_a = np.fromiter(S0.values(), dtype = int)
#comm.Gather(S0_a,S0Tab,root=0)

#HC_iter = np.array([HC_iter],dtype=int)
#comm.Gather(HC_iter,IterTab,root=0)

#if Me == 0:
    #EbTab.resize(NbP)
    #SbTab.resize(NbP, nd)
    #S0Tab.resize(NbP, nd)
    #IterTab.resize(NbP, nd)
    
    #HC_best_E = np.amax(EbTab)
    #HC_best_E_arg = np.argmax(EbTab)

    #HC_best_S0 = S0Tab[HC_best_E_arg]
    #HC_best_Sb = SbTab[HC_best_E_arg]
    
    #EbTab = np.zeros(NbP*1,dtype=np.float64)
    #SbTab = np.zeros(NbP*nd,dtype=int)
    #S0Tab = np.zeros(NbP*nd,dtype=int)
    #IterTab = np.zeros(NbP*1,dtype=int)

#Parallel Hill Climbing
S0 = GC.generateS0()
PHC_eb, PHC_sb,PHC_iter = HC.HillClimbing(S0, S0['nb_it'], "flops")

PHC_eb = np.array([PHC_eb],dtype=np.float64)
comm.Gather(PHC_eb,EbTab,root=0)

PHC_sb_a = np.fromiter(PHC_sb.values(), dtype = int)
comm.Gather(Sb,SbTab,root=0)

S0_a = np.fromiter(S0.values(), dtype = int)
comm.Gather(S0_a,S0Tab,root=0)
comm.Gather(S0,S0Tab,root=0)

PHC_iter = np.array([PHC_iter],dtype=int)
comm.Gather(PHC_iter,IterTab,root=0)

if Me == 0:
    EbTab.resize(NbP)
    SbTab.resize(NbP, nd)
    S0Tab.resize(NbP, nd)
    IterTab.resize(NbP, nd)

    PHC_best_E = np.amax(EbTab)
    PHC_best_E_arg = np.argmax(EbTab)

    PHC_best_S0 = S0Tab[PHC_best_E_arg]
    PHC_best_Sb = SbTab[PHC_best_E_arg]

    EbTab = np.zeros(NbP*1,dtype=np.float64)
    SbTab = np.zeros(NbP*nd,dtype=int)
    S0Tab = np.zeros(NbP*nd,dtype=int)
    IterTab = np.zeros(NbP*1,dtype=int)

#Greedy
GR_eb, GR_sb, GR_iter = GR.parallel_greedy(S0, S0['nb_it'], NbP, Me)

GR_eb = np.array([GR_eb],dtype=np.float64)
comm.Gather(GR_eb,EbTab,root=0)

GR_sb_a = np.fromiter(GR_sb.values(), dtype = int)
comm.Gather(GR_sb_a,SbTab,root=0)

S0_a = np.fromiter(S0.values(), dtype = int)
comm.Gather(S0_a,S0Tab,root=0)

GR_iter = np.array([GR_iter],dtype=int)
comm.Gather(GR_iter,IterTab,root=0)

if Me == 0:
    EbTab.resize(NbP)
    SbTab.resize(NbP, nd)
    S0Tab.resize(NbP, nd)
    IterTab.resize(NbP, nd)

    GR_best_E = np.amax(EbTab)
    GR_best_E_arg = np.argmax(EbTab)

    GR_best_S0 = S0Tab[GR_best_E_arg]
    GR_best_Sb = SbTab[GR_best_E_arg]

    EbTab = np.zeros(NbP*1,dtype=np.float64)
    SbTab = np.zeros(NbP*nd,dtype=int)
    S0Tab = np.zeros(NbP*nd,dtype=int)
    IterTab = np.zeros(NbP*1,dtype=int)
#Printing results
if Me == 0:
    print(20*"=" + "Hill Climbing" + 20*"=")
    print("Best energy: " + str(HC_eb) + " Best Solution: " + str(HC_sb))
    print("Iter: " + str(HC_iter))

    print(20*"=" + "Parallel HC" + 20*"=")
    print("Best energy: " + str(PHC_best_E) + " Best Solution: " + str(PHC_best_Sb))
    print("Iter: " + str(PHC_iter))
    
    print(20*"="+ "Parallel Greedy" + 20*"=")
    print("Best energy: " + str(GR_best_E) + " Best Solution: " + str(GR_best_Sb))
    print("Iter: " + str(GR_iter))




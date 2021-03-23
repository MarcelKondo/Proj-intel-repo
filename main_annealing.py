from mpi4py import MPI
import numpy as np
import time
import os
import deploy_greedy_v3
import parallel_tabu
#import mpi_HillClimbing
#import run_simul_annealing_mpi
import sys, getopt, argparse
import HillClimbing as HC
import main_parallel_HC as mpi_HC
import general_config as GC
from server_content.automated_compiling_tabu import define_copiler_settings
import simulated_annealing as SA

comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

comm.barrier()

T0 = 80
IterMax = 10
la = 0.8

if Me == 0:
    print("PE: ", Me, "/",NbP,": all processes started")   

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
def execute(S0, args):


    eb, Sb,iter = SA.SimulatedAnnealing(S0, args.IterMax, T0, la)

    Eb = np.array([eb],dtype=np.float64)
    comm.Gather(Eb,EbTab,root=0)

    comm.Gather(np.array([x for x in Sb.values()]),SbTab,root=0)
    comm.Gather(np.array([x for x in S0.values()]),S0Tab,root=0)

    Iter = np.array([iter],dtype=int)
    comm.Gather(Iter,IterTab,root=0)
    #Print results
    if Me == 0:
        nd = GC.GetNbDim()
        EbTab.resize(NbP)
        S0Tab.resize(NbP, nd)
        SbTab.resize(NbP, nd)
        IterTab.resize(NbP, nd)
    comm.barrier()
    time.sleep(1)
    if Me == 0:
        best_E = np.amax(EbTab)
        best_E_arg = np.argmax(EbTab)
        best_S0 = S0Tab[best_E_arg]
        best_Sb = SbTab[best_E_arg]
    
    
    else:
        best_E = None
        best_S0 = None
        best_Sb = None
    return best_E,best_S0, best_Sb
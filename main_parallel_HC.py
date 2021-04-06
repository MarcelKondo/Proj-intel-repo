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
import general_config as GC
from server_content.automated_compiling_tabu import define_copiler_settings


comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()


comm.barrier()

if Me == 0:
    print("PE: ", Me, "/",NbP,": all processes started")   
    


def execute(S0, args):
    if (Me == 0):
        nd = GC.GetNbDim()
        EbTab = np.zeros(NbP*1,dtype=np.float64)
        SbTab = np.zeros(NbP*nd,dtype=int)
        S0Tab = np.zeros(NbP*nd,dtype=int)
        IterTab = np.zeros(NbP*1,dtype=int)
    else:
        EbTab   = None     
        SbTab   = None
        S0Tab   = None
        IterTab = None

    PHC_eb, PHC_sb,PHC_iter = HC.HillClimbing(S0, args.iter_max, "flops")
    print('\n')
    print('\n')
    print('\n')
    S0.pop('simdType',None) #simdType not int and useless at this point
    PHC_sb.pop('simdType',None) #simdType not int and useless at this point

    PHC_eb = np.array([PHC_eb],dtype=np.float64)
    comm.Gather(PHC_eb,EbTab,root=0)

    PHC_sb_a = np.fromiter(PHC_sb.values(), dtype = int)
    comm.Gather(PHC_sb_a,SbTab,root=0)

    S0_a = np.fromiter(S0.values(), dtype = int)
    comm.Gather(S0_a,S0Tab,root=0)
            
    PHC_iter = np.array([PHC_iter],dtype=int)
    comm.Gather(PHC_iter,IterTab,root=0)
    #Print results
    if Me == 0:
        nd = GC.GetNbDim()
        EbTab.resize(NbP)
        SbTab.resize(NbP, nd)
        S0Tab.resize(NbP, nd)
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

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


param_list = ['tblock1','tblock2','tblock3'] #parse CLI

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

#HillClimbing
HC_eb, HC_sb, HC_iter = HC.HillClimbing(S0, S0['nb_it'], "flops")

#Greedy
GR_eb, GR_sb, GR_iter = GR.parallel_greedy(S0, S0['nb_it'], NbP, Me)


#Printing results
print(20*"Hill Climbing"*20)
print("Best energy: " + str(HC_eb) + " Best Solution: " + str(HC_sb))
print("Iter: " + HC_iter)

print(20*"Parallel Greedy"*20)
print("Best energy: " + str(GR_eb) + " Best Solution: " + str(GR_sb))
print("Iter: " + GR_iter)




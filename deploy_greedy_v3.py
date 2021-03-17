import os
import sys
import time
import numpy as np
import math
import random as rd
from mpi4py import MPI


from numpy.core.arrayprint import SubArrayFormat

from server_content.automated_compiling_tabu import find_number, define_exec_param, define_copiler_settings, Cost

comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()


param_space = {
    'n1' : [256, 500, 0],
    'n2' : [256, 500, 0],
    'n3' : [256, 500, 0],
    'nb_threads' : [4, 10, 0],
    'nb_it' : [10, 20, 0],
    'tblock1' : [32, 32, 16],
    'tblock2' : [32, 32, 16],
    'tblock3' : [32, 32, 16],
    'simdType' : ["avx512"]
}

def get_neighbourhood(S):
    LNgbh = []
  
    for param in param_space.keys():
        if param == 'simdType':
            p_idx = param_space[param].index(S[param])
            S1 = S.copy()
            if p_idx + 1 < len(param_space[param]):
                S1[param] = param_space[param][p_idx + 1]
                LNgbh.append(S1)
            
            S2 = S.copy()
            if p_idx - 1 >= 0:
                S2[param] = param_space[param][p_idx - 1]
                LNgbh.append(S2)
        else:
            for k in range(1,5):
                Skp = S.copy()
                Skm = S.copy()

                Skp[param] += k*param_space[param][2]
                if Skp[param] < param_space[param][1]:
                    LNgbh.append(Skp)

                Skm[param] -= k*param_space[param][2]
                if Skm[param] > 0:
                    LNgbh.append(Skm)
    return LNgbh
 
  
def find_best(LNgbh, NbP, Me): #à paralléliser
    e = 0
    S = None
    n = len(LNgbh)
    q = n//NbP
    rest = n%NbP
    j = Me*q
    if j==n-rest:
      liste_p = [LNgbh[i+j] for i in range(rest)]
    else:
      liste_p = [LNgbh[i+j] for i in range(q)]
    for Sp in liste_p:
       ep = Cost(Sp)
            #print('ep',ep)
       if ep > e :
          S = Sp
          e = ep
    #print("me e",Me,e)
    mi= [e,Me]
    e,rank = comm.allreduce(mi,op=MPI.MAXLOC)
    #print("BROADCAST")
    S= comm.bcast(S, root=rank)
    return S, e

def parallel_greedy(S0,IterMax,NbP, Me):  
   
    Sb = S0
    #print("so",S0)
    eb = Cost(Sb)
    iter = 0
    NewBetterS = True

    S = Sb
    e = eb
    LNgbh = get_neighbourhood(S)

    while iter < IterMax and NewBetterS:
        S,e = find_best(LNgbh, NbP, Me) 
        if e > eb:
            #print("Eb GLOBAL TROUVÉ")
            Sb = S
            eb = e
            LNgbh = get_neighbourhood(Sb)
            #print(len(LNgbh))
        else:
            NewBetterS = False
        iter += 1
        print(20*"=","NEW ITERATION",20*"=")
    print("[TG] END")
    
    return eb,Sb,iter

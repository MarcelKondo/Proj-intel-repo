import os
import sys
from itertools import combinations
import time
import numpy as np
import math
import random as rd
from mpi4py import MPI

import general_config as GC
from numpy.core.arrayprint import SubArrayFormat

from server_content.automated_compiling_tabu import find_number, define_exec_param, define_copiler_settings, Cost

comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()



def get_neighbourhood(S):
    LNgbh = []
    for param in param_space.keys():
        if param == 'simdType':
            None
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


def nghbrhd_other(S):
    LNgbh =[]
    keys = ['n1','n2','n3','tblock1','tblock2','tblock3']
    triplets = list(combinations(keys,3))  #toutes combinaisons de triplets possibles
    for _ in range(5):
        liste_params = rd.sample(triplets,6) # on n'en garde que 6 pour chaque itération
        for params in liste_params:
            S_new = S.copy()
            for param in params:
                rd_bool = bool(rd.getrandbits(1)) #random boolean
                k = rd.randint(1,10)
                if S_new[param]+k*param_space[param][2] < param_space[param][1] and S_new[param] - k*param_space[param][2] >0:
                    S_new[param] += k*param_space[param][2]*rd_bool
                    S_new[param] -= k*param_space[param][2]*(1-rd_bool)

            LNgbh.append(S_new)
    return LNgbh 

def guided_cost(S,Sb,LNgbh):#,Levol):
  LNloc= get_neighbourhood(S)
  e = Cost(S)
  lda = 0.5
  for X in LNloc:
    if X in LNgbh:
      e+=lda*0.5#le poids
    elif X ==Sb:
      e+=lda*0.6#le poids central est un petit plus important
    
  return e

def fifo_add(Sb, L_tabu, tabu_size):
    if len(L_tabu)==tabu_size:
        L_tabu.pop(0)
    L_tabu.append(Sb)
    return L_tabu

def find_best(LNgbh, L_tabu, NbP, Me,Sb): #à paralléliser
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
        if Sp not in L_tabu:
            ep = guided_cost(Sp,Sb,LNgbh)
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





def parallel_tabu_greedy(S0,IterMax,tabu_size, NbP, Me):  
    """#S0: initial solution
    #IterMax: max nb of iteration
    # tabu_size: length of Tabu list for "Tabu List" method"""
    print(f"[TG] STARTED OPTIMISATION : itermax:{IterMax}, tabu_size:{tabu_size}")
    Sb = S0
    #print("so",S0)
    eb = Cost(Sb)
    iter = 0
    NewBetterS = True

    S = Sb
    e = eb
    LNgbh = GC.nghbrhd_other(S)
    L_tabu = [Sb]

    while iter < IterMax and NewBetterS:
        S,e = find_best(LNgbh, L_tabu, NbP, Me,Sb) 
        if e > eb:
            #print("Eb GLOBAL TROUVÉ")
            Sb = S
            eb = e
            L_tabu = fifo_add(Sb, L_tabu, tabu_size)
            LNgbh = GC.nghbrhd_other(Sb)
            #print(len(LNgbh))
        else:
            NewBetterS = False
        iter += 1
        print(20*"=","NEW ITERATION",20*"=")
    print("[TG] END")
    
    return eb,Sb,iter

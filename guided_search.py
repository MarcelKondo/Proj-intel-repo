import os
import sys
import time
import numpy as np
import math
import random as rd

from numpy.core.arrayprint import SubArrayFormat

from server_content.automated_compiling_tabu import find_number, define_exec_param, define_copiler_settings, Cost

param_space = {
    'n1' : [256, 500, 0],
    'n2' : [256, 500, 1],
    'n3' : [256, 500, 1],
    'nb_threads' : [4, 10, 1],
    'nb_it' : [10, 20, 1],
    'tblock1' : [32, 32, 0],
    'tblock2' : [32, 32, 0],
    'tblock3' : [32, 32, 0],
    'simdType' : ["avx512"]
}

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

def find_best(LNgbh, L_tabu, NbP, Me): #à paralléliser
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
        S,e = find_best(LNgbh, L_tabu, NbP, Me) 
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

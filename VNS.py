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



def get_neighbourhood(S, param):
    LNgbh = []
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

def parallel_greedy_VNS(S0,IterMax, NbP, Me, param):  
    """#S0: initial solution
    #IterMax: max nb of iteration
    # tabu_size: length of Tabu list for "Tabu List" method"""
  
    Sb = S0
    #print("so",S0)
    eb = Cost(Sb)
    iter = 0
    NewBetterS = True

    S = Sb
    e = eb
    LNgbh = get_neighbourhood(S, param)


    while iter < IterMax and NewBetterS:
        S,e = find_best(LNgbh, L_tabu, NbP, Me) 
        if e > eb:
            #print("Eb GLOBAL TROUVÉ")
            Sb = S
            eb = e
            LNgbh = get_neighbourhood(Sb, param)
            #print(len(LNgbh))
        else:
            NewBetterS = False
        iter += 1
        print(20*"=","NEW ITERATION",20*"=")
    print("Greedy END")
    
    return eb,Sb,iter

exploring_param = ['n1', 'n2', 'n3', 'tblock1', 'tblock2', 'tblock3']
  
def run_VNS_greedy(S0, IterMax, NbP, Me, exploring_param):
    tot_iter = 0
    for param in exploring_param:
        e,S,iter = parallel_greedy_VNS(S0, IterMax, NbP, Me, param)  #Recherche de la meilleure solution locale pour le seul paramètre variable "param"
        print("optimisation selon {0} donne une vitesse de {1} en {2} itérations. Nouvelle valeur optimale de {0} = {3}".format(param,e,iter,S[param])
        S0 = S  #Récupère le neighbordhood localement optimal où seulement param a changé.
        tot_iter += iter
     print(20*"__", "VNS greedy terminé")
     return e,S,tot_iter      
              
if __name__ == "__main__":
    eb,Sb,tot_iter = run_VNS_greedy(S0, IterMax, NbP, Me, exploring_param)
    print(f"Best score: {eb}, Solution: {str(Sb)}, total_Iters: {tot_iter}")
             

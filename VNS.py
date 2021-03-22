import os
import sys
from itertools import combinations
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
    'n1' : [256, 500, 16],
    'n2' : [256, 500, 4],
    'n3' : [256, 500, 4],
    'nb_threads' : [4, 10, 0],
    'nb_it' : [10, 20, 0],
    'tblock1' : [32, 32, 16],
    'tblock2' : [32, 32, 4],
    'tblock3' : [32, 32, 4],
    'simdType' : ["avx512"]
}

def get_neighbourhood(S, param):
    LNgbh = []
    if param == 'simdType':
        None
    else:
        for k in range(1,3):
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

def parallel_greedy_VNS(S0, IterMax, NbP, Me, param):  
    """#S0: initial solution
    #IterMax: max nb of iteration
    # param: param of param_space that changes in the neighborhood during exploration"""
  
    Sb = S0
    #print("so",S0)
    eb = Cost(Sb)
    iter = 0
    NewBetterS = True
    S = Sb
    e = eb
    LNgbh = get_neighbourhood(S, param)
    
    while iter < IterMax and NewBetterS:
        S,e = find_best(LNgbh,  NbP, Me) 
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

def run_VNS_greedy(S0, IterMax, NbP, Me, exploring_param):          #exploring_param = ['n1', 'n2', 'n3', 'tblock1', 'tblock2', 'tblock3']
    tot_iter = 0
    S0 = S0
    for param in exploring_param:
        e,S,iter = parallel_greedy_VNS(S0, IterMax, NbP, Me, param)  #Recherche de la meilleure solution locale pour le seul paramètre variable "param"
        S0 = S                 #Récupère dans S le neighbordhood localement optimal où seulement param a changé.
        print(20*"__", "optimisation selon {0} donne une vitesse de {1} en {2} itérations. Nouvelle valeur optimale de {0} = {3}".format(param,e,iter,S[param])) 
        tot_iter += iter
    print(20*"__", "VNS greedy terminé")
    return e,S,tot_iter      
    
   

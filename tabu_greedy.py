import os
import sys
import time
import numpy as np
import math
import random as rd

from numpy.core.arrayprint import SubArrayFormat

from server_content.automated_compiling import find_number, define_exec_param, define_copiler_settings, Cost

param_space = {
    'n1' : [256, 500, 16],
    'n2' : [256, 500, 1],
    'n3' : [256, 500, 1],
    'nb_threads' : [4, 10, 1],
    'nb_it' : [10, 20, 1],
    'tblock1' : [32, 50, 1],
    'tblock2' : [32, 50, 1],
    'tblock3' : [32, 50, 1],
    'simdType' : ["sse"]
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
            S1 = S.copy()
            S1[param] += param_space[param][2]
            if S1[param] < param_space[param][1]:
                LNgbh.append(S1)
            
            S2 = S.copy()
            S2[param] -= 1
            if S2[param] >= param_space[param][0]:
                LNgbh.append(S2)
    
    return LNgbh

def find_best(LNgbh, L_tabu):
    e = 10e9
    S = None
    for Sp in LNgbh:
        if Sp not in L_tabu:
            ep = Cost(Sp)
            if ep > e :
                S = Sp
                e = ep
    return S, e

def fifo_add(Sb, L_tabu, tabu_size):
    if len(L_tabu)==tabu_size:
        L_tabu.pop(0)
    L_tabu.append(Sb)
    return L_tabu



def tabu_greedy(S0,IterMax,tabu_size):  
    """#S0: initial solution
    #IterMax: max nb of iteration
    # tabu_size: length of Tabu list for "Tabu List" method"""
    print(f"[TG] STARTED OPTIMISATION : itermax:{IterMax}, tabu_size:{tabu_size}")
    Sb = S0
    eb = Cost(Sb)
    iter = 0
    NewBetterS = True

    S = Sb
    e = eb
    LNgbh = get_neighbourhood(S)
    L_tabu = [Sb]

    while iter < IterMax and NewBetterS:
        S,e = find_best(LNgbh, L_tabu)
        if e > eb:
            Sb = S
            eb = e
            L_tabu = fifo_add(Sb, L_tabu, tabu_size)
        else:
            NewBetterS = False
        iter += 1
    print("[TG] END")
    
    return eb,Sb,iter

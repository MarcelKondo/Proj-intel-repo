import os
import sys
import time
import numpy as np
import math
import random as rd
import general_config as GC
from server_content.automated_compiling_tabu import find_number, define_exec_param, define_copiler_settings, Cost

param_space = {
    'n1' : [256, 257, 16],
    'n2' : [256, 257, 1],
    'n3' : [256, 257, 1],
    'nb_threads' : [4, 5, 1],
    'nb_it' : [10, 11, 1],
    'tblock1' : [16, 97, 16],
    'tblock2' : [10, 100, 1],
    'tblock3' : [10, 100, 1]
}

param_space_categorical = {
    #'simdType' : ["sse"]
}

def GetNbDim():
    return len(param_space) + len(param_space_categorical)

def generateS0():
    #rd.seed(seed)
    S0 = {}
    for param in param_space.keys():
        lmin = param_space[param][0]
        lmax = param_space[param][1]
        delta = param_space[param][2]
        grid_size = int((lmax-lmin)/delta)
        if grid_size ==  0 :
            grid_size = 1 #case where gridsize is 0
        pos = rd.randint(0,grid_size-1)
        val = lmin + pos * delta
        S0[param] = val
        assert val % delta == 0, f"Random S0 value not in grid - S0[{param}] = {val}"

    for param in param_space_categorical.keys():
        param_vals = param_space_categorical[param]
        grid_size = len(param_vals)
        pos = rd.randint(0, grid_size-1)
        S0[param] = param_vals[pos]

    return S0

def get_neighbourhood(S):
    LNgbh = []
  
    for param in param_space.keys():
        S1 = S.copy()
        S1[param] += param_space[param][2]
        if S1[param] < param_space[param][1]:
            LNgbh.append(S1)
        
        S2 = S.copy()
        S2[param] -= param_space[param][2]
        if S2[param] >= param_space[param][0]:
            LNgbh.append(S2)
    
    for param in param_space_categorical.keys():
        p_idx = param_space_categorical[param].index(S[param]) #we'll take the order of the list to define the neighbourhood
        S1 = S.copy()
        if p_idx + 1 < len(param_space_categorical[param]):
            S1[param] = param_space_categorical[param][p_idx + 1]
            LNgbh.append(S1)
        
        S2 = S.copy()
        if p_idx - 1 >= 0:
            S2[param] = param_space_categorical[param][p_idx - 1]
            LNgbh.append(S2)

    return LNgbh




def SimulatedAnnealing(S0,IterMax,T0,la):  #ltl unused in SA
    #SO: initial solution
    #IterMax: max nb of iteration
    #T0: initial temperature for "simulated annealing"
    #la: T = T*la: temperature evolution law for "simulated annealing"
    #ltl: length of Tabu list for "Tabu List" method
    #simIdx: version of the simulator (cost function)
    print(f"[SMA] STARTED OPTIMISATION : itermax:{IterMax}, T0:{T0}, la:{la}")
    Sb = S0
    eb = Cost(Sb)
    iter = 0
    
    T = T0
    S = Sb
    e = eb
    LNgbh = GC.get_neighbourhood(S)
    while iter < IterMax:
        if iter%10 == 0:
            print(f"Iteration #{iter}")
        k = rd.randrange(len(LNgbh))
        Sp = LNgbh[k]
        ep = Cost(Sp)
        if ep > e or rd.random() < np.exp(-(ep - e)/T):
            S = Sp
            e = ep
            LNgbh = GC.get_neighbourhood(S)
            if (e > eb):
                Sb = S
                eb = e
        T = la*T
        iter += 1
    print("[SMA] END")
    
    return eb,Sb,iter
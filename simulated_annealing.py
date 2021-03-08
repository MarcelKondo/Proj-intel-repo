import os
import sys
import time
import numpy as np
import math
import random as rd

from server_content.automated_compiling import find_number, define_exec_param, define_copiler_settings, Cost

param_space = {
    'n1' : [256, 257, 1],
    'n2' : [256, 257, 1],
    'n3' : [256, 257, 1],
    'nb_threads' : [4, 5, 1],
    'nb_it' : [100, 101, 1],
    'tblock1' : [32, 33, 1],
    'tblock2' : [32, 33, 1],
    'tblock3' : [32, 33, 1],
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




def SimulatedAnnealing(S0,IterMax,T0,la,ltl,simIdx):  #ltl unused in SA
  #SO: initial solution
  #IterMax: max nb of iteration
  #T0: initial temperature for "simulated annealing"
  #la: T = T*la: temperature evolution law for "simulated annealing"
  #ltl: length of Tabu list for "Tabu List" method
  #simIdx: version of the simulator (cost function)
  Sb = S0
  eb = Cost(Sb)
  iter = 0
  
  T = T0
  S = Sb
  e = eb
  LNgbh = get_neighbourhood(S)
  while iter < IterMax:
    k = rd.randrange(len(LNgbh))
    Sp = LNgbh[k]
    ep = Cost(Sp, simIdx)
    if ep < e or rd.random() < np.exp(-(ep - e)/T):
      S = Sp
      e = ep
      LNgbh = get_neighbourhood(S)
      if (e < eb):
        Sb = S
        eb = e
    T = la*T
    iter += 1

  #local search                                                       TO DO

  #Note: At the beginning: 
  #  T = t0
  #
  #Note: At the end: update of temperature, and increase of the iteration 
  #  T = la*T
  #  iter += 1

  #return best Energy, best Solution, and nb of iter
  return eb,Sb,iter
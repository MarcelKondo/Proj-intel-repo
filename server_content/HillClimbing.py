import os
import subprocess 
import math
import random as rd
import numpy as np

import automated_compiling as autcom

NbDim = 8                       # 5 elts in a solution (n1,n2,n3,nb_t,nb_it,tblk1,tblk2,tblk3) (opt and simdType not used yet)
size = 256                            
lmin = [32, 32, 32, 1, 100, 16, 16, 16]
lmax = [272, 272, 272, 8, 100, 80, 80, 80]

def rand_multiple(fac, a, b):
    """Returns a random multiple of fac between a and b."""
    min_multi = math.ceil(float(a) / fac)
    max_multi = math.floor(float(b) / fac)
    return fac * rd.randint(min_multi, max_multi)

def generateS0():
 # rd.seed(Me+1000*(1+SeedInc))
  S0 = np.empty(NbDim,dtype=np.int)
  for i in range(NbDim):
    if(i == 0 or i == 5  or i == 6 or i == 7 ):
        S0[i] = rand_multiple(16, lmin[i], lmax[i]+1)
    elif (i == 4):
        S0[i] = 100
    else:
        S0[i] = rd.randrange(lmin[i],lmax[i]+1,1)
  return(S0)


def Neighborhood(S, param_indices):                                                
    LNgbh = []
  
    for i in param_indices:
        S1 = S.copy()
        if(i == 0 or i == 5 or i == 6 or i == 7):
            S1[i] += 16
        else:
            S1[i] += 4
        if S1[i] <= lmax[i]:
            LNgbh.append(S1)
            print(LNgbh)
        
        S2 = S.copy()
        if(i == 0 or i == 5 or i == 6 or i == 7):
            S2[i] -= 16
        else:
            S2[i] -= 4
        if S2[i] >= lmin[i]:
            LNgbh.append(S2)
            print(LNgbh)
    
    return LNgbh

def HillClimbing(S0,IterMax,param_list,cost_type):  #T0, la, ltl unused in HC
    #SO: initial solution
    #IterMax: max nb of iteration
    #T0: initial temperature for "simulated annealing"
    #la: T = T*la: temperature evolution law for "simulated annealing"
    #ltl: length of Tabu list for "Tabu List" method
    #simIdx: version of the simulator (cost function)
    Sb = S0
    eb = autcom.Cost(Sb, cost_type = "flops")
    iter = 0
    
    #local search
    S = Sb
    LNgbh = Neighborhood(S, param_list)
    print(LNgbh)
    while iter < IterMax and len(LNgbh): #BetterSolFound:
        k = rd.randrange(len(LNgbh))
        Sk = LNgbh.pop(k)
        print("New parameters" + str(Sk))
        ek = autcom.Cost(Sk, cost_type = "flops")
        if ek < eb:
            Sb = Sk
            eb = ek
            LNgbh = Neighborhood(Sb, param_list)
        iter += 1
    
    #return best Energy, best Solution, and nb of iter
    return eb,Sb,iter

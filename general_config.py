import os
import sys
import time
import numpy as np
import math
import random as rd
from itertools import combinations

param_space = {
    'n1' : [160, 1000, 16],
    'n2' : [160, 1000, 4],
    'n3' : [160, 1000, 4],
    'nb_threads' : [32, 32, 0],
    'nb_it' : [10, 200, 10],
    'tblock1' : [16, 80, 16],
    'tblock2' : [16, 80, 4],
    'tblock3' : [16, 80, 4]
}

param_space_categorical = {
    #'simdType' : ["sse"]
}

neighbourhood = None

def GetNbDim():
    return len(param_space) + len(param_space_categorical)

def generateS0():
    #rd.seed(seed)
    S0 = {}
    for param in param_space.keys():
        lmin = param_space[param][0]
        lmax = param_space[param][1]
        delta = param_space[param][2]
        if delta == 0:
            grid_size = 1
        else:
            grid_size = int((lmax-lmin)/delta)
       # if grid_size ==  0 :
       #     grid_size = 1 #case where gridsize is 0
        pos = rd.randint(0,grid_size-1)
        val = lmin + pos * delta
        S0[param] = val
        if delta != 0:
            assert val % delta == 0, f"Random S0 value not in grid - S0[{param}] = {val}"

    for param in param_space_categorical.keys():
        param_vals = param_space_categorical[param]
        grid_size = len(param_vals)
        pos = rd.randint(0, grid_size-1)
        S0[param] = param_vals[pos]

    return S0

def define_usedParameters(param_list):
    '''Define parameters that will vary'''
    param_keys = list(param_space.keys())
    for param in param_list:
        if param in param_keys:
            param_keys.remove(param)
    
    for elem in param_keys:
        param_space[elem][2] = 0
    return 

def define_neighbourhood(nbgh_name):
    '''Define the type of neighbourhood to use'''
    global neighbourhood
    print(nbgh_name)
    if nbgh_name == "basic":
        neighbourhood = get_neighbourhood_basic
    else:
        neighbourhood = nghbrhd_other

def get_neighbourhood(S):
    '''neighbourhood that methods will implement'''

    LNgbh = neighbourhood(S)
    return LNgbh

def get_neighbourhood_basic(S):
    print("Basic neighbourhood")
    LNgbh = []
    print(param_space)
    for param in param_space.keys():
        if param_space[param][2] != 0:
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

def nghbrhd_other(S):
    LNgbh =[]
    print("another neighbourhood")
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
               
def ngh_other_more_local(S):
    LNgbh =[]
    print("another neighbourhood")
    keys = ['n1','n2','n3','tblock1','tblock2','tblock3']
    triplets = list(combinations(keys,3))  #toutes combinaisons de triplets possibles
    for _ in range(5):
        liste_params = rd.sample(triplets,6) # on n'en garde que 6 pour chaque itération
        for params in liste_params:
            S_new = S.copy()
            for param in params:
                rd_bool = bool(rd.getrandbits(1)) #random boolean
                k = rd.randint(1,2)
                if S_new[param]+k*param_space[param][2] < param_space[param][1] and S_new[param] - k*param_space[param][2] >0:
                    S_new[param] += k*param_space[param][2]*rd_bool
                    S_new[param] -= k*param_space[param][2]*(1-rd_bool)

            LNgbh.append(S_new)
    return LNgbh
               

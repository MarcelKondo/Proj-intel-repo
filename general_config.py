import os
import sys
import time
import numpy as np
import math
import random as rd


param_space = {
    'n1' : [32, 300, 16],
    'n2' : [32, 300, 4],
    'n3' : [32, 300, 4],
    'nb_threads' : [1, 8, 1],
    'nb_it' : [10, 50, 1],
    'tblock1' : [16, 80, 16],
    'tblock2' : [16, 80, 4],
    'tblock3' : [16, 80, 4]
}

param_space_categorical = {
    #'simdType' : ["sse"]
}

def GetNbDim():
    return len(param_space) + len(param_space_categorical)

def define_usedParameters(param_list):
    '''Define parameters that will vary'''
    param_keys = list(param_space.keys())
    for param in param_list:
        if param in param_keys:
            param_keys.remove(param)
    
    for elem in param_keys:
        param_space[elem][2] = 0
    return 

def get_neighbourhood(S):
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
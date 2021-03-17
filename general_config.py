import os
import sys
import time
import numpy as np
import math
import random as rd


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

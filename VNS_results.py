from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import os
import deploy_greedy_v3
import parallel_tabu
import sys, getopt, argparse
import HillClimbing as HC
import main_parallel_HC as main_HC
import main_greedy as main_greedy
import main_tabu_greedy as main_tabu_greedy
import main_annealing as main_SA
import general_config as GC
import simulated_annealing as SA
import function_VNS_local_methods as VNS_LM
import function_local_methods as LM
from server_content.automated_compiling_tabu import define_copiler_settings


comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


#Parameters to change

args = {
    'S0': [512, 512, 512, 32, 10, 32, 32, 32],
    'method': "GR",
    'param_list': ['n1','n2','n3','tblock1','tblock2','tblock3'], 
    'iter_max': 10,
    'tabu_size': 8,
    'opt': 3,
    'simdType': "avx512",
    'neighbourhood': "basic"
}

args = dotdict((args))

define_copiler_settings(opLevel=args.opt, simdType=args.simdType, version="dev13")



neighbs = ['LM','VNS_LM']  #Deux types de neighborhood
best_flops = dict()
average_flops = dict()
best_times = dict()
average_times = dict()

for ng in neighbs:
    best_flops[ng] = 0
    best_times[ng] = 0
    average_flops[ng] = 0
    average_times[ng] = 0

imax = 2 # nb runs
for i in range(0,imax):

    for ng in neighbs:
#         args.neighbourhood = ng
        if ng == 'LM':
            current_E, current_Sb, current_S0, current_dt = LM.execute(args)
        else: # ng =='VNS_LM':
             current_E, current_Sb, current_S0, current_dt = VNS_LM.execute(args)    
            
        if Me == 0:
            if current_E > best_flops[ng]:
                    best_flops[ng] = current_E
                    best_times[ng] = current_dt

        average_flops[ng] += current_E
        average_times[ng] += current_dt
        
    if Me == 0:
        print('\n')
        print('best result so far: ')
        print('\n')
        print(f'best_flops: {best_flops}')
        print(f'best_times: {best_times}')

if Me == 0:
    average_flops = {key:value/imax for key, value in average_flops.items()}
    average_times = {key:value/imax for key, value in average_times.items()}
    print('\n')
    print('\n')
    print('\n')
    print(f'best_flops: {best_flops}')
    print(f'best_times: {best_times}')
    print(f'average_flops: {average_flops}')
    print(f'average_times: {average_times}')
    df = pd.DataFrame({'Gflops': list(best_flops.values()), 'Execution time (s)': list(best_times.values()), 'Average speed': list(average_flops.values()), 'Average time': list(average_times.values())}, index = neighbs)
    print(df)
    #ax = df.plot.bar(rot=0)
    df.to_csv(r'~/Proj-intel-repo/VNS_greedy.csv', index = True, header=True)

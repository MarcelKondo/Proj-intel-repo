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
import function_VNS_local_methods as run_LM
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
    'method': "all",
    'param_list': ['n1','n2','n3','tblock1','tblock2','tblock3'], 
    'iter_max': 10,
    'tabu_size': 8,
    'opt': 3,
    'simdType': "avx512",
    'neighbourhood': "basic"
}

args = dotdict((args))

define_copiler_settings(opLevel=args.opt, simdType=args.simdType, version="dev13")



methods = ['GR']
best_flops = dict()
average_flops = dict()
best_times = dict()
average_times = dict()

for method in methods:
    best_flops[method] = 0
    best_times[method] = 0
    average_flops[method] = 0
    average_times[method] = 0

imax = 2 # nb runs
for i in range(0,imax):

    for method in methods:
        args.method = method
        if method == 'HC' or method == 'SA':
            if Me == 0:
                current_E,current_Sb, current_S0,current_dt = run_LM.execute(args)

                if current_E > best_flops[method]:
                    best_flops[method] = current_E
                    best_times[method] = current_dt
        else:
            current_E,current_Sb, current_S0,current_dt = run_LM.execute(args)
            if Me == 0:
                if current_E > best_flops[method]:
                        best_flops[method] = current_E
                        best_times[method] = current_dt

        average_flops[method] += current_E
        average_times[method] += current_dt
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
    df = pd.DataFrame({'Gflops': list(best_flops.values()), 'Execution time (s)': list(best_times.values()), 'Average speed': list(average_flops.values()), 'Average time': list(average_times.values())}, index = methods)
    print(df)
    #ax = df.plot.bar(rot=0)
    df.to_csv(r'~/Proj-intel-repo/VNS_greedy.csv', index = True, header=True)

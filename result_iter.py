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
import function_local_methods as run_LM
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
itertest= [1,2,5,10,20,40,80,160,320,500]
args = {
    'S0': [256, 256, 256, 32, 10,32,32,32],
    'method': "all",
    'param_list': ['n1','n2','n3','tblock1','tblock2','tblock3'], 
    'iter_max': 5,
    'tabu_size': 8,
    'opt': 3,
    'simdType': "avx512",
    'neighbourhood': "other"
}

args = dotdict((args))



define_copiler_settings(opLevel=args.opt, simdType=args.simdType, version="dev13")



methods = ['HC', 'PHC', 'GR', 'TGR', 'SA']
best_energies = dict()
best_times = dict()
dict_iter = dict()


print('initial solution: ')
print(args.S0)

for method in methods:
    best_energies[method] = []
    best_times[method] = []
    dict_iter[method]=[]

imax = 1 # nb runs
for nbiter in itertest:
    for i in range(0,imax):
        args["iter_max"]= nbiter
        for method in methods:
            args.method = method
            if method == 'HC' or method == 'SA':
                if Me == 0:
                    current_E,current_Sb, current_S0,current_dt = run_LM.execute(args)
                    best_energies[method].append(current_E)
                    best_times[method].append(current_dt)
                    dict_iter[method].append(args.iter_max)
            else:
                current_E,current_Sb, current_S0,current_dt = run_LM.execute(args)

                if Me == 0:
                    best_energies[method].append(current_E)
                    best_times[method].append(current_dt)
                    dict_iter[method].append(args.iter_max)

        if Me == 0:
            print('\n')
            print('best result so far: ')
            print('\n')
            print(f'best_energies: {best_energies}')
            print(f'best_times: {best_times}')
            
        

if Me == 0:
    print('\n')
    print('\n')
    print('\n')
    print(f'best_energies: {best_energies}')
    print(f'best_times: {best_times}')
    print(f'dict_iter: {dict_iter}')
    print("S0",S0)
    df = pd.DataFrame({'Gflops': best_energies, 'Execution time (s)': best_times,'iteration ordre':dict_iter}, index = methods)
    print(df)
    #ax = df.plot.bar(rot=0)
    df.to_csv('/usr/users/cpust75/cpust75_5/Proj-intel-repo/Iteration.csv', index = True, header=True)
    
    
    
    
    

    
    
    
    
    
    

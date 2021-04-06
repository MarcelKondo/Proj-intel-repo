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

args = {
    'S0': [512, 512, 512, 8, 10,32,32,32],
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



methods = ['HC', 'PHC', 'GR', 'TGR', 'SA']

best_energies = dict()
average_energies = dict()
worst_energies = dict()


best_times = dict()
average_times = dict()
worst_times = dict()




neighbourhoods = ['basic', 'others', 'local']


for neighbourhood in neighbourhoods:

    for method in methods:
        best_energies[method] = 0
        best_times[method] = 0

        average_energies[method] = 0
        average_times[method] = 0

        worst_energies[method] = 100000
        worst_times[method] = 1000000

    args.neighbourhood = neighbourhood
    print('\n')
    print('\n')
    print(f'Neighbourhood: {args.neighbourhood}')
    for method in methods:
        args.method = method
        print('\n')
        print('\n')
        print(f'NEW INITIAL SOLUTION: {args.S0}')
        print('\n')
        print('\n')
        if method == 'HC' or method == 'SA':
            if Me == 0:
                current_E,current_Sb, current_S0,current_dt = run_LM.execute(args)

                average_energies[method] += current_E
                average_times[method] += current_dt
                if current_E > best_energies[method]:
                    best_energies[method] = current_E
                    best_times[method] = current_dt

                if current_E < worst_energies[method]:
                    worst_energies[method] = current_E
                    worst_times[method] = current_dt
        else:
            current_E,current_Sb, current_S0,current_dt = run_LM.execute(args)

            if Me == 0:
                average_energies[method] += current_E
                average_times[method] += current_dt
                if current_E > best_energies[method]:
                    best_energies[method] = current_E
                    best_times[method] = current_dt
                if current_E < worst_energies[method]:
                    worst_energies[method] = current_E
                    worst_times[method] = current_dt

        
    if Me == 0:
        print('\n')
        print('best result so far: ')
        print('\n')
        print(f'best_energies: {best_energies}')
        print(f'best_times: {best_times}')
        print(f'worst_energies: {worst_energies}')
        print(f'worst_times: {worst_times}')

    if Me == 0:
        #average_energies = {key:value/imax for key, value in average_energies.items()}
        #average_times = {key:value/imax for key, value in average_times.items()}
        print('\n')
        print('\n')
        print('\n')
        print(f'best_energies: {best_energies}')
        print(f'best_times: {best_times}')
        print(f'average_energies: {average_energies}')
        print(f'average_times: {average_times}')
        df = pd.DataFrame({'Best Gflops': list(best_energies.values()), 'Best Exec time (s)': list(best_times.values()), 'Average Gflops': list(average_energies.values()), 'Average time': list(average_times.values()), 'Worst Gflops': list(worst_energies.values()), 'Worst time': list(worst_times.values())}, index = methods)
        print(df)
        #ax = df.plot.bar(rot=0)
        path = '~/Proj-intel-repo/Results_' + neighbourhood + '.csv'
        df.to_csv(path, index = True, header=True)



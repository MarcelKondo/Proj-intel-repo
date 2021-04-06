from mpi4py import MPI
import numpy as np
import json
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

dict_tot = dict()




#Parameters to change
i=0
while i!=1: 
    print("=== /!\ NOUVEAU POINT S0 /!\ ===")
    print('Point S0 numéro ',i+1,'/5')
    S0 = GC.generateS0()
    print(S0)
    speeds = dict()
    times = dict()
    solutions = dict()
    
    for tabu_size in np.arange(1,3,2):
        print("=== /!\ NOUVELLE TABU_SIZE POUR S0 /!\ ===")
        print("Tabu_size :",tabu_size)
        args = {
            'S0': S0,
            'method': "TGR",
            'param_list': ['n1','n2','n3','tblock1','tblock2','tblock3'], 
            'iter_max': 10,
            'tabu_size': tabu_size,
            'opt': 3,
            'simdType': "avx512",
            'neighbourhood': "basic"
        }

        args = dotdict((args))
        define_copiler_settings(opLevel=args.opt, simdType=args.simdType, version="dev13")

        current_E,current_Sb, current_S0,current_dt = run_LM.execute(args)
        print('==================== PRINTING TO FOLLOW ====================')
        print('Speed: ',current_E)
        print('Tabu_size: ', args.tabu_size)
        
        speeds[int(tabu_size)] = current_E
        times[int(tabu_size)] = current_dt
        solutions[int(tabu_size)] = current_Sb
        
    dict_tot[int(i+1)] = {'point init':S0, 'vitesses':speeds, 'temps':times, 'Solutions':solutions}
    print(dict_tot[int(i+1)])
    i+=1
print(dict_tot)

a_file = open('/usr/users/cpust75/cpust75_14/Proj-Intel/Proj-intel-repo/tabu_size.json', "w")
json.dump(dict_tot, a_file)
a_file.close()

#df = pd.DataFrame({'Gflops': list(best_energies.values()), 'Execution time (s)': list(best_times.values()), 'Average speed': list(average_energies.values()), 'Average time': list(average_times.values())}, index = np.arange(1,12,2))
#print(df)
#ax = df.plot.bar(rot=0)
#df.to_csv(r'~/Proj-intel-repo/tabu_size.csv', index = True, header=True)

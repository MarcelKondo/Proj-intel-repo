from mpi4py import MPI
import numpy as np
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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = {
    'S0': [256, 256, 256, 4, 10,32,32,32],
    'method': "HC",
    'param_list': ['n1','n2','n3'], 
    'iter_max': 10,
    'tabu_size': 8,
    'opt': 3,
    'simdType': "avx512",
    'neighbourhood': "basic"
}

args = dotdict((args))

best_E,best_Sb, best_S0,dt = run_LM.execute(args)

print(f'energy: {best_E}')


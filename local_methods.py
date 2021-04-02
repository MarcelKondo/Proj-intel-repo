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
import general_config as GC
from server_content.automated_compiling_tabu import define_copiler_settings


comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

comm.barrier()
if Me == 0:
    print("PE: ", Me, "/",NbP,": all processes started")   
    
        
S0 = {
    'n1' : 256,
    'n2' : 256,
    'n3' : 256,
    'nb_threads' : 4,
    'nb_it' : 10,
    'tblock1' : 32,
    'tblock2' : 32,
    'tblock3' : 32,
    #'simdType' : 'avx512'
}

param_space = {
    'n1' : [256, 1024, 16],
    'n2' : [256, 1024, 16],
    'n3' : [256, 1024, 16],
    'nb_threads' : [4, 10, 0],
    'nb_it' : [10, 20, 0],
    'tblock1' : [32, 128, 16],
    'tblock2' : [32, 128, 4],
    'tblock3' : [32, 128, 4],
    #'simdType' : ["avx512"]
}

def parse():
    parser = argparse.ArgumentParser('Geral Config')
    parser.add_argument('-S0', '--S0', nargs='+', type=int)
    parser.add_argument('-method', '--method', metavar='', help="specify the method used (HC, PHC, GR, TGR, SA")
    parser.add_argument('-pl', '--param_list', nargs="+", help ="parameters to change")
    parser.add_argument('-itm', '--iter_max', type=int, metavar='',help='IterMax')
    parser.add_argument('-ts', '--tabu_size', type=int, metavar='',help='tabu_size')
    parser.add_argument('-opt', '--opt', default = 3, type=int, metavar='',help='Compiler optimization mode')
    parser.add_argument('-simdType', '--simdType', default = "avx512", metavar='',help='Compiler optimization mode')
    parser.add_argument('-ngbr', '--neighbourhood', default = "basic", metavar='',help="Specify the type of neighbourhood used")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

   
    args = parse()

    define_copiler_settings(opLevel=args.opt, simdType=args.simdType)

    GC.define_usedParameters(args.param_list)
    GC.define_neighbourhood(args.neighbourhood)

    if(args.S0 == None):
        S0 = GC.generateS0()
    else:
        assert args.S0[0] % 16 == 0, "n1 must be divisble by 16"
        assert args.S0[5] % 16 == 0, "tblock1 must be divisible by 16"

        params = ['n1','n2','n3','nb_threads','nb_it','tblock1','tblock2','tblock3']
        for i in range(len(args.S0)):
            S0[params[i]] = args.S0[i]

    if args.simdType != None:
        S0['simdType'] = args.simdType

    if(args.method == "HC"):
        #Execute only HillClimbing
        print(f"Executing only {args.method}")
        if(Me == 0):
            print(20*"=","HILL CLIMBING",20*"=")
            eb_HC, Sb_HC, iters_HC = HC.HillClimbing(S0, args.iter_max, "flops")

            print(20*"=","HILL CLIMBING",20*"=")
            print("eb_HC",eb_HC,"Sb_HC",Sb_HC,"iters_HC", iters_HC)
            print('\n')
    elif(args.method == "PHC"):
        #Execute only Parallel_HC
        print(f"Executing only {args.method}")
        best_E, best_S0, best_Sb = main_HC.execute(S0,args)

        print("\n")
        print("========================= Best Parameters ======================")
        print("Parallel HillClimbing")
        print("\n")
        print("Best performance (Gflops) " + str(best_E))
        print("Initial solution " + str(best_S0))
        print("Optimal solution " + str(best_Sb))


    elif (args.method == "GR"):
        #Execute only Greedy
        print(f"Executing only {args.method}")

        best_E, best_S0, best_Sb = main_greedy.execute(S0, args)
        print("\n")
        print("========================= Best Parameters ======================")
        print("Parallel Greedy")
        print("Best performance (Gflops) " + str(best_E))
        print("Initial Solution " + str(best_S0))
        print("Optimal solution " + str(best_Sb))
        print("\n")

    
    elif (args.method == "TGR"):
        #Execute only Tabu Greedy
        print(f"Executing only {args.method}")
        
        best_E, best_S0, best_Sb = main_tabu_greedy.execute(S0, args)
        print("\n")
        print("========================= Best Parameters ======================")
        print("Tabu Greedy")
        print("Best performance (Gflops) " + str(best_E))
        print("Initial Solution " + str(best_S0))
        print("Optimal solution " + str(best_Sb))
        print("\n")
    else:
        #Execute all methods
        print(f"Executing all methods")

        #HillClimbing
        if Me == 0:
            eb_HC, Sb_HC, iters_HC = HC.HillClimbing(S0, args.iter_max, "flops")
        time.sleep(1)

        S0['simdType'] = args.simdType
        #Parallel HillClimbing
        best_E_PHC, best_S0_PHC, best_Sb_PHC = main_HC.execute(S0,args)
        S0['simdType'] = args.simdType

        #Greedy
        best_E_GR, best_S0_GR, best_Sb_GR = main_greedy.execute(S0, args)
        S0['simdType'] = args.simdType

        #Tabu Greedy
        best_E_TGR, best_S0_TGR, best_Sb_TGR = main_tabu_greedy.execute(S0, args)
        S0['simdType'] = args.simdType
        if Me == 0:
            print("\n")
            print("========================= Best Parameters ======================")
            print("Hill Climbing")
            print("\n")
            print("Best performance (Gflops) " + str(eb_HC))
            print("Initial solution " + str(S0))
            print("Optimal solution " + str(Sb_HC))

            print("\n")
            print("========================= Best Parameters ======================")
            print("Parallel HillClimbing")
            print("\n")
            print("Best performance (Gflops) " + str(best_E_PHC))
            print("Initial solution " + str(best_S0_PHC))
            print("Optimal solution " + str(best_Sb_PHC))

            print("\n")
            print("========================= Best Parameters ======================")
            print("Parallel Greedy")
            print("Best performance (Gflops) " + str(best_E_GR))
            print("Initial Solution " + str(best_S0_GR))
            print("Optimal solution " + str(best_Sb_GR))
            print("\n")

            print("\n")
            print("========================= Best Parameters ======================")
            print("Parallel Greedy")
            print("Best performance (Gflops) " + str(best_E_TGR))
            print("Initial Solution " + str(best_S0_TGR))
            print("Optimal solution " + str(best_Sb_TGR))
            print("\n")






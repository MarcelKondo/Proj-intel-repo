from mpi4py import MPI
import numpy as np
import time
import os
import deploy_greedy_v3
import parallel_tabu
#import mpi_HillClimbing
#import run_simul_annealing_mpi
import sys, getopt, argparse
import HillClimbing as HC
import main_parallel_HC as main_HC
import main_greedy as main_greedy
import general_config as GC
from server_content.automated_compiling_tabu import define_copiler_settings


define_copiler_settings(opLevel=3, simdType="avx512")
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
    #parser.add_argument('-S0', '--S0',  )
    parser.add_argument('-method', '--method', metavar='', help="specify the method used (HC, PHC, GR, TGR, SA")
    parser.add_argument('-pl', '--param_list', nargs="+", help ="parameters to change")
    parser.add_argument('-itm', '--iter_max', type=int, metavar='',required=True,help='IterMax')
    parser.add_argument('-ts', '--tabu_size', type=int, metavar='',required=True,help='tabu_size')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    #print('ARGS', sys.argv[1:])
    args = parse()
    GC.define_usedParameters(args.param_list)

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
        print("Best Energy " + str(best_E))
        print("Optimal solution " + str(best_Sb))


    elif (args.method == "GR"):
        #Execute only Greedy
        print(f"Executing only {args.method}")

        best_E, best_S0, best_Sb = main_greedy.execute(S0, args)
        print("\n")
        print("========================= Best Parameters ======================")
        print("Parallel Greedy")
        print("Best Energy " + str(best_E))
        print("Initial Solution " + str(best_S0))
        print("Optimal solution " + str(best_Sb))
        print("\n")

    
    elif (args.method == "TGR"):
        #Execute only Tabu Greedy
        print(f"Executing only {args.method}")
        if (Me == 0):
            nd = GC.GetNbDim()
            EbTab = np.zeros(NbP*1,dtype=np.float64)
            SbTab = np.zeros(NbP*nd,dtype=int)
            S0Tab = np.zeros(NbP*nd,dtype=int)
            IterTab = np.zeros(NbP*1,dtype=int)
        else:
            EbTab   = None     
            SbTab   = None
            S0Tab   = None
            IterTab = None

        TGR_eb, TGR_sb, TGR_iter = parallel_tabu.parallel_tabu_greedy(S0,args.iter_max,args.tabu_size, NbP, Me)

        TGR_eb = np.array([TGR_eb],dtype=np.float64)
        comm.Gather(TGR_eb,EbTab,root=0)

        TGR_sb_a = np.fromiter(TGR_sb.values(), dtype = int)
        comm.Gather(TGR_sb_a,SbTab,root=0)
        
        TGR_iter = np.array([TGR_iter],dtype=int)
        comm.Gather(TGR_iter,IterTab,root=0)
        #Print results
        if Me == 0:
            nd = GC.GetNbDim()
            EbTab.resize(NbP)
            SbTab.resize(NbP, nd)
            IterTab.resize(NbP, nd)
        comm.barrier()
        time.sleep(1)
        if Me == 0:
            best_E = np.amax(EbTab)
            best_E_arg = np.argmax(EbTab)
            best_Sb = SbTab[best_E_arg]
            print("\n")
            print("========================= Best Parameters ======================")
            print("Tabu Greedy")
            print("\n")
            
            print("Best Energy " + str(best_E))
            print("Optimal solution " + str(best_Sb))
            print("PE: ", Me, "/",NbP," bye!")



    # #run hill climbing
    # print(20*"=","HILL CLIMBING",20*"=")
    # eb_HC, Sb_HC, iters_HC = HC.HillClimbing(S0, args.iter_max, "flops")
    # #ebhill, Sbhill,itershill= mpi_HillClimbing.HillClimbing(S0,args.iter_max,param_list,cost_type):
    # #cmd = 'mpirun -np 4 -map-by ppr:2:socket -bind-to socket python3 run_parallel_tabu.py'
    # #os.system(cmd)
    
    
    # #run greedy
    # print(20*"=","GREEDY",20*"=")
    # eb, Sb, iters = deploy_greedy_v3.parallel_greedy(S0,args.iter_max, NbP, Me)
    # print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")
    
    # #run tabu_greedy
    # print(20*"=","TABU",20*"=")
    # #print('ARGS', sys.argv[1:])
    # args = parse()
    # ebtab, Sbtab, iterstab = parallel_tabu.parallel_tabu_greedy(S0,args.iter_max,args.tabu_size, NbP, Me)
    # print(f"Best score: {ebtab}, Solution: {str(Sbtab)}, Iters: {iterstab}")

    # #run annealing
    # #print(20*"=","ANNEALING",20*"=")
    # #cmd = 'mpirun -np 4 -map-by ppr:2:socket -bind-to socket python3 run_simul_annealing_mpi.py'
    # #os.system(cmd)
    
    
    # Lmethod= [[eb, Sb, iters],[ebtab, Sbtab, iterstab]]


    # print(20*"=","GREEDY",20*"=")
    # print("eb",eb,"Sb",Sb,"iters", iters)
    # print('\n')

    # print(20*"=","TABU",20*"=")
    # print("ebtab",ebtab,"Sbtab",Sbtab,"iterstab", iterstab)
    # print('\n')

    # print(20*"=","HILL CLIMBING",20*"=")
    # print("eb_HC",eb_HC,"Sb_HC",Sb_HC,"iters_HC", iters_HC)
    # print('\n')

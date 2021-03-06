from mpi4py import MPI
import os
import deploy_greedy_v3
import parallel_tabu
#import mpi_HillClimbing
#import run_simul_annealing_mpi
import sys, getopt, argparse

from server_content.automated_compiling_tabu import define_copiler_settings


define_copiler_settings(opLevel=3, simdType="avx512")
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

S0 = {
    'n1' : 368,
    'n2' : 228,
    'n3' : 292,
    'nb_threads' : 4,
    'nb_it' : 10,
    'tblock1' : 48,
    'tblock2' : 33,
    'tblock3' : 28,
    'simdType' : "avx512"
}


def parse():
    parser = argparse.ArgumentParser('Greedy Parallel')
    parser.add_argument('-itm', '--iter_max', type=int, metavar='',required=True,help='IterMax')
    parser.add_argument('-ts', '--tabu_size', type=int, metavar='',required=True,help='tabu_size')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #print('ARGS', sys.argv[1:])
    args = parse()
    #run hill climbing
    print(20*"=","HILL CLIMBING",20*"=")
    #ebhill, Sbhill,itershill= mpi_HillClimbing.HillClimbing(S0,args.iter_max,param_list,cost_type):
    #cmd = 'mpirun -np 4 -map-by ppr:2:socket -bind-to socket python3 run_parallel_tabu.py'
    #os.system(cmd)
    
    
    #run greedy
    print(20*"=","GREEDY",20*"=")
    eb, Sb, iters = deploy_greedy_v3.parallel_greedy(S0,args.iter_max, NbP, Me)
    print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")
    
    #run tabu_greedy
    print(20*"=","TABU",20*"=")
    print('ARGS', sys.argv[1:])
    args = parse()
    ebtab, Sbtab, iterstab = parallel_tabu.parallel_tabu_greedy(S0,args.iter_max,args.tabu_size, NbP, Me)
    print(f"Best score: {ebtab}, Solution: {str(Sbtab)}, Iters: {iterstab}")

    #run annealing
    #print(20*"=","ANNEALING",20*"=")
    #cmd = 'mpirun -np 4 -map-by ppr:2:socket -bind-to socket python3 run_simul_annealing_mpi.py'
    #os.system(cmd)
    
    
    Lmethod= [[eb, Sb, iters],[ebtab, Sbtab, iterstab]]
    if eb>ebtab:
        print("Greedy simple is better")
        print("eb",eb,"Sb",Sb,"iters", iters)
        print("greedy tabu result")
        print("ebtab",ebtab,"Sbtab",Sbtab,"iterstab", iterstab)
    else:
        print("greedy tabu is better")
        print("ebtab",ebtab,"Sbtab",Sbtab,"iterstab", iterstab)
        print("Greedy simple result")
        print("eb",eb,"Sb",Sb,"iters", iters)

from mpi4py import MPI
import parallel_tabu
import sys, getopt, argparse
import general_config as GC
from server_content.automated_compiling_tabu import define_copiler_settings


define_copiler_settings(opLevel=3, simdType="avx512")
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

# S0 = GC.generateS0()
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

param_space = {
    'n1' : [256, 1024, 16],
    'n2' : [256, 1024, 4],
    'n3' : [256, 1024, 4],
    'nb_threads' : [4, 10, 0],
    'nb_it' : [10, 20, 0],
    'tblock1' : [32, 128, 16],
    'tblock2' : [32, 128, 1],
    'tblock3' : [32, 128, 1],
    'simdType' : ["avx512"]
}


def parse():
    parser = argparse.ArgumentParser('Greedy Parallel Tabu')
    parser.add_argument('-itm', '--iter_max', type=int, metavar='',required=True,help='IterMax')
    parser.add_argument('-ts', '--tabu_size', type=int, metavar='',required=True,help='tabu_size')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #print('ARGS', sys.argv[1:])
    args = parse()
    eb, Sb, iters = parallel_tabu.parallel_tabu_greedy(S0,args.iter_max,args.tabu_size, NbP, Me)
    print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")



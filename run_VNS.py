from mpi4py import MPI
from VNS import run_VNS_greedy
import sys, getopt, argparse

from server_content.automated_compiling_tabu import define_copiler_settings


define_copiler_settings(opLevel=3, simdType="avx512")
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

param_space = {
    'n1' : [256, 500, 16],
    'n2' : [256, 500, 4],
    'n3' : [256, 500, 4],
    'nb_threads' : [4, 10, 0],
    'nb_it' : [10, 20, 0],
    'tblock1' : [32, 32, 16],
    'tblock2' : [32, 32, 4],
    'tblock3' : [32, 32, 4],
    'simdType' : ["avx512"]
}

S0 = {
    'n1' : 256,
    'n2' : 256,
    'n3' : 256,
    'nb_threads' : 4,
    'nb_it' : 10,
    'tblock1' : 32,
    'tblock2' : 32,
    'tblock3' : 32,
    'simdType' : "avx512"
}

exploring_param = ['n1', 'n2', 'n3', 'tblock1', 'tblock2', 'tblock3']

def parse():
    parser = argparse.ArgumentParser('Greedy VNS')
    parser.add_argument('-itm', '--iter_max', type=int, metavar='',required=True,help='IterMax')
#     parser.add_argument('-param', '--exploring_param', type=int, metavar='',required=True,help='exploring_param')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #print('ARGS', sys.argv[1:])
    args = parse()
    eb, Sb, tot_iter = run_VNS_greedy(S0, args.iter_max, NbP, Me, exploring_param)
    print(f"Best score: {eb}, Solution: {str(Sb)}, total_Iters: {tot_iter}")
             

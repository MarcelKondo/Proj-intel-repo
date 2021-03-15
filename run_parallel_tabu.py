from mpi4py import MPI
import parallel_tabu
from server_content.automated_compiling import define_copiler_settings


define_copiler_settings(opLevel=3, simdType="avx512")
comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

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
IterMax = 5
tabu_size = input("Tabu size?")
eb, Sb, iters = parallel_tabu.parallel_tabu_greedy(S0,IterMax,tabu_size, NbP, Me)
print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")

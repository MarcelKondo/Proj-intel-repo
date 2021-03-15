from mpi4py import MPI
import parallel_tabu
import sys, getopt

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


def main(argv):
   IterMax = ''
   tabu_size = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["iter_max=","tabu_size="])
   except getopt.GetoptError:
      print('run_parallel_tabu.py -itm <IterMax> -ts <tabu_size>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('run_parallel_tabu.py -itm <IterMax> -ts <tabu_size>')
         sys.exit()
      elif opt in ("-itm", "--iter_max"):
         IterMax = arg
      elif opt in ("-ts", "--tabu_size"):
         tabu_size = arg
   eb, Sb, iters = parallel_tabu.parallel_tabu_greedy(S0,int(IterMax),int(tabu_size), NbP, Me)
   print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")

if __name__ == "__main__":
   main(sys.argv[1:])



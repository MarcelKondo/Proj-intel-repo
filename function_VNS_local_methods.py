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
from server_content.automated_compiling_tabu import define_copiler_settings


comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()

comm.barrier()
if Me == 0:
    print("PE: ", Me, "/",NbP,": all processes started")   
    
param_space = {
    'n1' : [256, 1024, 16],
    'n2' : [256, 1024, 16],
    'n3' : [256, 1024, 16],
    'nb_threads' : [32, 32, 0],
    'nb_it' : [10, 200, 5],
    'tblock1' : [32, 128, 16],
    'tblock2' : [32, 128, 4],
    'tblock3' : [32, 128, 4],
    #'simdType' : ["avx512"]
}
    
 
def execute(args):
    S0 = {
    'n1' : 256,
    'n2' : 256,
    'n3' : 256,
    'nb_threads' : 32,
    'nb_it' : 10,
    'tblock1' : 32,
    'tblock2' : 32,
    'tblock3' : 32,
    #'simdType' : 'avx512'
    }

  GC.define_neighbourhood(args.neighbourhood)  
    
  if(args.S0 == None):      #Si on ne renseigne pas de point de départ, on en genère un aléatoirement.
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
          t1 = time.time()  
          print(20*"=","HILL CLIMBING",20*"=")
          tot_iter_HC = 0
          best_Sb = S0
          for param in args.param_list:
            define_usedParameters([param])
            best_E, best_Sb, iters_HC = HC.HillClimbing(best_Sb, args.iter_max, "flops")
            best_Sb = listToDict(best_Sb)
            tot_iter_HC += iters_HC
            best_S0 = S0
          dt = time.time()-t1
          print(20*"=","HILL CLIMBING",20*"=")
          print("eb_HC",best_E,"Sb_HC",best_Sb,"iters_HC", tot_iter_HC)
          print("Execution time {:.3f}".format(dt))
          print('\n')
          
  elif(args.method == "PHC"):
      #Execute only Parallel_HC
      t1 = time.time()
      print(f"Executing only {args.method}")
      best_Sb = S0
      for param in args.param_list:
        define_usedParameters([param])
        best_E, best_S0, best_Sb = main_HC.execute(best_Sb,args)
        best_Sb = listToDict(best_Sb)
      dt = time.time()-t1
      print("\n")
      print("========================= Best Parameters ======================")
      print("Parallel HillClimbing")
      print("\n")
      print("Best performance (Gflops) " + str(best_E))
      print("Initial solution " + str(best_S0))
      print("Optimal solution " + str(best_Sb))
      print("Execution time {:.3f}".format(dt))


  elif (args.method == "GR"):
      #Execute only Greedy
      t1 = time.time()
      print(f"Executing only {args.method}")
      best_Sb = S0
      for param in args.param_list:
        define_usedParameters([param])
        best_E, best_S0, best_Sb = main_greedy.execute(best_Sb, args)
        best_Sb = listToDict(best_Sb)
      dt = time.time() - t1
      print("\n")
      print("========================= Best Parameters ======================")
      print("Parallel Greedy")
      print("Best performance (Gflops) " + str(best_E))
      print("Initial Solution " + str(best_S0))
      print("Optimal solution " + str(best_Sb))
      print("Execution time {:.3f}".format(dt))
      print("\n")


  elif (args.method == "TGR"):
      #Execute only Tabu Greedy
      t1 = time.time()
      print(f"Executing only {args.method}")
      best_Sb = S0
      for param in args.param_list:
        define_usedParameters([param])
        best_E, best_S0, best_Sb = main_tabu_greedy.execute(best_Sb, args)
        best_Sb = listToDict(best_Sb)
      dt_TGR = time.time() - t1
      print("\n")
      print("========================= Best Parameters ======================")
      print("Tabu Greedy")
      print("Best performance (Gflops) " + str(best_E))
      print("Initial Solution " + str(best_S0))
      print("Optimal solution " + str(best_Sb))
      print("Execution time {:.3f}".format(dt_TGR))
      print("\n")
    
  elif (args.method == "SA"):
      #Execute only Tabu Greedy
      t1 = time.time()
      print(f"Executing only {args.method}")
      print(20*"=","SIMU",20*"=")
      best_Sb = S0
      if Me == 0:
          for param in args.param_list:
              define_usedParameters([param])
              best_E, best_Sb, iters_SA = SA.SimulatedAnnealing(best_Sb, args.iter_max, 80, 0.8)
              best_Sb = listToDict(best_Sb)
              best_S0 = S0
          dt = time.time() - t1
          print(20*"=","SIMU",20*"=")
          print("best E",best_E,"Best Sb",Best_Sb,"iters_SA", iters_SA)
          print("Execution time {:.3f}".format(dt))
          print('\n')
          
  return best_E, best_Sb, best_S0, dt

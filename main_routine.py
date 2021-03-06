
import HillClimbing as HC
import sys 
import general_config as GC
#S0 = [256, 256, 256, 4, 10, 32, 32, 32]

S0 = {
    'n1' : 256,
    'n2' : 256,
    'n3' : 256,
    'nb_threads' : 4,
    'nb_it' : 10,
    'tblock1' : 32,
    'tblock2' : 32,
    'tblock3' : 32,
}

param_list = ['n1','n1','n1']
GC.define_usedParameters(param_list)
#nb_args = len(sys.argv) - 1 #first element is the script name
#args = sys.argv[1:]
#args = [int(i) for i in args]
#if(nb_args != 0):
#    S0[0:nb_args] = args
eb,sb,iter = HC.HillClimbing(S0, S0['nb_it'], "flops")
print("\n")
print("========================= Best Parameters ======================")
print("HillClimbing")
print("\n")
print("Best Energy " + str(eb))
print("Initial solution " + str(S0))
print("Best solution " + str(sb))

print("Finished")


import HillClimbing as HC
import sys 

S0 = [256, 256, 256, 4, 10, 32, 32, 32]

nb_args = len(sys.argv) - 1 #first element is the script name
args = sys.argv[1:]
args = [int(i) for i in args]
if(nb_args != 0):
    S0[0:nb_args] = args
eb,sb,iter = HC.HillClimbing(S0, 10, [5, 6, 7], "flops")
print("\n")
print("========================= Best Parameters ======================")
print("HillClimbing")
print("\n")
print("Best Energy " + str(eb))
print("Initial solution " + str(S0))
print("Best solution " + str(sb))

print("Finished")

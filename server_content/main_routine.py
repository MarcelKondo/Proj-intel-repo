
import HillClimbing as HC
import sys 

S0 = [256, 256, 256, 4, 10, 32, 32, 32]

nb_args = len(sys.argv) - 1 #first element is the script name
args = sys.argv[1:]
args = [int(i) for i in args]
if(nb_args != 0):
    S0[0:nb_args] = args
res = HC.HillClimbing(S0, 20, [5, 6, 7], "flops")
print(res)
print("Finished")

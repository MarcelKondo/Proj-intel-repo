
import HillClimbing as HC
import sys 

S0 = [240, 240, 240, 3, 100, 32, 32, 32]

nb_args = len(sys.argv) - 1
if(nb_args != 0):
    S0[0:nb_args] = sys.argv
res = HC.HillClimbing(S0, 20, [0, 1, 2], "flops")
print(res)
print("Finished")

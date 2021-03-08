
import HillClimbing as HC

S0 = [256, 256, 256, 3, 100, 32, 32, 32]
res = HC.HillClimbing(S0, 20, [0, 1, 2], "flops")
print(res)
print("Finished")
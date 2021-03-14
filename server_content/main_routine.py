
import HillClimbing as HC

S0 = [240, 240, 240, 3, 100, 32, 32, 32]
res = HC.HillClimbing(S0, 20, [0, 1, 2], "flops")
print(res)
print("Finished")

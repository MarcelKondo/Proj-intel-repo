import simulated_annealing

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
T0 = 100
IterMax = 5
la = 0.8

eb, Sb, iters = simulated_annealing.SimulatedAnnealing(S0, IterMax, T0, la)

print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")
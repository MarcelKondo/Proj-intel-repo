import tabu_greedy

S0 = {
    'n1' : 256,
    'n2' : 256,
    'n3' : 256,
    'nb_threads' : 4,
    'nb_it' : 100,
    'tblock1' : 32,
    'tblock2' : 32,
    'tblock3' : 32,
    'simdType' : "sse"
}
IterMax = 5
tabu_size = 2
eb, Sb, iters = tabu_greedy.tabu_greedy(S0,IterMax,tabu_size)

print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")

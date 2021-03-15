import tabu_greedy
from server_content.automated_compiling import define_copiler_settings

define_copiler_settings(opLevel=3, simdType="avx512")

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
IterMax = 5
tabu_size = 2
eb, Sb, iters = tabu_greedy.tabu_greedy(S0,IterMax,tabu_size)
print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")

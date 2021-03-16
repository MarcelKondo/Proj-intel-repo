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

def parse():
    parser = argparse.ArgumentParser('Greedy Parallel Tabu')
    parser.add_argument('-itm', '--iter_max', type=int, metavar='',required=True,help='IterMax')
    parser.add_argument('-ts', '--tabu_size', type=int, metavar='',required=True,help='tabu_size')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #print('ARGS', sys.argv[1:])
    args = parse()
    eb, Sb, iters = tabu_greedy.tabu_greedy(S0,args.iter_max,args.tabu_size)
    print(f"Best score: {eb}, Solution: {str(Sb)}, Iters: {iters}")

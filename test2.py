import sys, getopt, argparse



def parse():
    parser = argparse.ArgumentParser("Teste")
    parser.add_argument('-S0', '--S0', nargs='+', type=int)
    parser.add_argument('-method', '--method', metavar='', help="specify the method used")
    parser.add_argument('-pl', '--param_list', nargs="+", help ="parameters to change")
    parser.add_argument('-itm', '--iter_max', type=int, metavar='',required=True,help='IterMax')
    parser.add_argument('-ts', '--tabu_size', type=int, metavar='',required=True,help='tabu_size')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    
    print(f'Iter: {args.iter_max}')
    print(f'Param_list: {args.param_list}')
    print(f'tabu_size: {args.tabu_size}')
    print(f'method: {args.method}')
    print(f'S0: {args.S0}')
    if(args.method == "HC"):
        print("HC")
    
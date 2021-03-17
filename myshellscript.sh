#!/bin/sh
#SBATCH --time=15

mpirun -np 4 -map-by ppr:2:socket -bind-to socket python3 run_multi.py -itm <IterMax> -ts <tabu_size>

# Méthodes employées
Run ``mpirun -np 4  -map-by ppr:2:socket -bind-to socket python3 local_methods.py -S0 <array_S0> -method <Method> -pl <param_list> -itm <IterMax> -ts <tabu_size>`` <br/>
Optional parameters:   <br/>
`-method` if `None` execute all methods <br/>
`-S0` if 'None' generate random S0 <br/>
<br/>
<br/>
Usage example: <br/>


## Greedy parallèle


## Tabu Greedy parallèle

Run ``mpirun -np 4 -map-by ppr:2:socket -bind-to socket python3 run_parallel_tabu.py -itm <IterMax> -ts <tabu_size>``

## SA parallèle

## HC parallèle

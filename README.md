# Méthodes employées
Run ``mpirun -np 4  -map-by ppr:2:socket -bind-to socket python3 local_methods.py -S0 <array_S0> -method <Method> -pl <param_list> -itm <IterMax> -ts <tabu_size> -opt <opt_option> -simdType <smid_type> -nbgr <nbgr_type>`` <br/>
Optional parameters:   <br/>
`-method` if `None` execute all methods. Available inputs: `HC`, `PHC`, `GR`, `TGR`, `ANE`<br/>
`-S0` if `None` generate random S0 <br/>
`-opt` if `None` default O3 <br/>
`-simdType` if `None` default avx512 <br/>
`-ngbr` if `None` default basic  <br/>
<br/>
<br/>
Usage example: <br/>

``mpirun -np 4  -map-by ppr:2:socket -bind-to socket python3 local_methods.py -S0 240 240 240 3 10 32 32 32 -method TGR -pl n1 n2 n3 -itm 10 -ts 8
``

## Greedy parallèle


## Tabu Greedy parallèle

Run ``mpirun -np 4 -map-by ppr:2:socket -bind-to socket python3 run_parallel_tabu.py -itm <IterMax> -ts <tabu_size>``

## SA parallèle

## HC parallèle

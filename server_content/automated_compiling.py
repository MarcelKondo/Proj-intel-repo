
import os
import subprocess
#import pandas as pd
#import numpy as np


def find_number(str):
    value = 0
    for t in str.split():
        try:
            value = float(t)
        except ValueError:
            pass
    return value

def define_exec_param(n1 = 256, n2 = 256, n3 = 256, nb_threads = 4, nb_it = 100, tblock1 = 32 , tblock2 = 32, tblock3 = 32, simdType = "sse"):
    res = subprocess.run("cd ~/Proj-Intel/Appli-iso3dfd/bin && iso3dfd_dev05_cpu_" + simdType + ".exe " + str(n1) + " " + str(n2) + " " + str(n3) + 
                         " " + str(nb_threads) + " " + str(nb_it) + " " + str(tblock1) + " " + str(tblock2) + " " + str(tblock3), shell=True,
                         stdout=subprocess.PIPE)
    res_str = str(res.stdout,'utf-8')
    print(res_str)

    res_str_lines = res_str.split('\n')
    
    flops = find_number(res_str_lines[-2])
    thrpt = find_number(res_str_lines[-3])
    time  = find_number(res_str_lines[-4])
    print(flops)
    print(thrpt)
    print(time)
    return [flops, thrpt, time]

def define_copiler_settings(opLevel, simdType):

    os.system("cd ~/Proj-Intel/Appli-iso3dfd/ && make -e OPTIMIZATION=\"-O"+ str(opLevel) + "\" -e simd=" + simdType + " last" )
    
print("V2 starting execution")
#d = {'n1' : n1, "n2" : n2, "n3" : n3, "nb_threads" : nb_threads, "nb_it": nb_it, "tblock1" : tblock1, 
     #"tblock2": tblock2, "tblock3": tblock3, "opLevel": opLevel, "simdType": simdType, "time": time, "throughput": thrpt, "flops": flops}
#df = pd.Dataframe(data=d)


def Cost(param, cost_type = "flops"):
    '''This function calculates only the cost using the exec parameters,
       none of the compilator parameters are treated in this case. '''
    list_values = define_exec_param(*param)

    if(cost_type == "flops"):
        e = list_values[0]
    else:
        e = 0
    return e

param = [256, 256 ,256, 4, 100, 32 ,32, 32, "sse"]
define_copiler_settings(opLevel = 3, simdType = "sse")
e = Cost(param, cost_type="flops")
print("e: " + str(e))
print("execution finished")

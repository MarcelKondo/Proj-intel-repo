import os
import subprocess

f_prefix = '../Appli-iso3dfd/'

def compile_program(args='-O2', file_name=''):
    os.system("(cd " + f_prefix + " && make " + args + " last)", shell=True)

def deploySUBP(strategy = 'socket', file_name=''):
    # - Generates the « allocated machines file »
    subprocess.run("sort -u $OAR_NODEFILE -o machines.txt", shell=True)
    # - Automatic counting of the number of allocated nodes
    file = open("machines.txt","r")
    liste = file.readlines()
    nbNodes = len(liste)
    # - Deployment of the application on all the allocated nodes
    #    according to the deployment rules and strategy
    print("NbNodes: " + str(nbNodes) + ", Strategy: " + strategy) 
    # - "socket" deployment strategy on Kyle
    if strategy == "socket":
        nbProcesses = 2*nbNodes
        res = subprocess.run("mpirun -np " + str(nbProcesses) + " -machinefile machines.txt" +
                            " -map-by ppr:1:socket -rank-by socket -bind-to socket" +
                            " python3 " + f_prefix + file_name + " 8",
                            shell=True,
                            stdout=subprocess.PIPE)      
    # - "core" deployment strategy on Kyle
    if strategy == "core":
        nbProcesses = 16*nbNodes
        res = subprocess.run("mpirun -np " + str(nbProcesses) + " -machinefile machines.txt" +
                            " -map-by ppr:1:core -rank-by core -bind-to core" +
                            " python3 " + f_prefix + file_name + " 1",
                            shell=True,
                            stdout=subprocess.PIPE)
    # - print MPI pgm output
    print(str(res.stdout,'utf-8'))
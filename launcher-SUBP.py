import subprocess

#----------------------------------------------------------------
# Deployment function: launch a MPI pgm on a set of cluster nodes
# + get the MPI pgm output and achieve a pretty print of the perf
#----------------------------------------------------------------
def deploySUBP(matSide,strategy):
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
                           " python3 matMulRing-Srr.py " + str(matSide) + " 8",
                         shell=True,
                         stdout=subprocess.PIPE)      
  # - "core" deployment strategy on Kyle
  if strategy == "core":
    nbProcesses = 16*nbNodes
    res = subprocess.run("mpirun -np " + str(nbProcesses) + " -machinefile machines.txt" +
                           " -map-by ppr:1:core -rank-by core -bind-to core" +
                           " python3 matMulRing-Srr.py " + str(matSide) + " 1",
                         shell=True,
                         stdout=subprocess.PIPE)
  # - print MPI pgm output
  print(str(res.stdout,'utf-8'))


#-----------------------------------------------------------------
# Main code
#-----------------------------------------------------------------

print("Deployment using Subprocess module")
deploySUBP(8192,"socket")
deploySUBP(8192,"core")




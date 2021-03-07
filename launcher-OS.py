import os

#----------------------------------------------------------------
# Deployment function: launch a MPI4PY pgm on a set of cluster nodes
# + autocompute the number of allocated nodes
#----------------------------------------------------------------
def deployOS(matSide,strategy):
   # - Generates the « allocated machines file »
   os.system("sort -u $OAR_NODEFILE -o machines.txt")
   # - Automatic counting of the number of allocated nodes
   file = open("machines.txt","r")
   lines = file.readlines()
   nbNodes = len(lines)
   # - Deployment of the application on all the allocated nodes 
   #    according to the deployment rules and strategy
   print("NbNodes: " + str(nbNodes) + ", Strategy: " + strategy) 
   # - "socket" deployment strategy on Kyle
   if strategy == "socket":
      nbProcesses = 2*nbNodes
      os.system("mpirun -np " + str(nbProcesses) + " -machinefile machines.txt" +
                " -map-by ppr:1:socket -rank-by socket -bind-to socket" +
                " python3 matMulRing-Srr.py " + str(matSide) + " 8")
   # - "core" deployment strategy on Kyle
   if strategy == "core":
      nbProcesses = 16*nbNodes
      os.system("mpirun -np " + str(nbProcesses) + " -machinefile machines.txt" +
                " -map-by ppr:1:core -rank-by core -bind-to core" +
                " python3 matMulRing-Srr.py " + str(matSide) + " 1")


#-----------------------------------------------------------------
# Main code
#-----------------------------------------------------------------

print("Deployment using OS module")
deployOS(8192,"socket")
deployOS(8192,"core")





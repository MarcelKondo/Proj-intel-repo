import numpy as np
from mpi4py import MPI
comm= MPI.COMM_WORLD
NbP= comm.Get_size()
Me = comm.Get_rank()
L=[1, 2, 3, 4, 5, 6, 7, 8]
n= len(L)   # size of neighborhood
q= n//NbP   # nombre moyen de voisinages traité par un processeur
rest=n%NbP  # 
j= Me*q     #
if j==n-rest:
    liste_p= [j+i for i in range(q+rest)]  #liste des indices du sous-voisinage
else:
    liste_p= [j+i for i in range(q)] #liste des indices du sous-voisinage

TabE= np.zeros(len(liste_p))
for i in liste_p:
        TabE[i-j]= (L[i]-4)**2
jp=0
for i in range(len(liste_p)):
    if TabE[i]<TabE[jp]:
        jp=i
E=TabE[jp]
Mi=[E,Me]
a,b = comm.allreduce(Mi, op=MPI.MAXLOC)
print("new E = {}, rank = {}".format(a,b))

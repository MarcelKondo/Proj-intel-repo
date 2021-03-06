import numpy as np
import subprocess
import glob
import itertools 
from os.path import basename, splitext
#exemple de commande: bin/iso3dfd_dev05_cpu_avx.exe 256 256 256 4 100 256 12 30
from mpi4py import MPI
comm= MPI.COMM_WORLD
NbP= comm.Get_size()
Me = comm.Get_rank()
kmax=5  #nombre max d'itération de recherche
NewbetterS= True #il existe un meilleur point dans le voisinnage
Sbest= [256,256,256]# point de départ de search le mieux est de séparer l'espace et parral les recherches 
#S=[x,y,z] les trois valeurs de position du point pour l'instant juste la size
r= 1 #rayon pour le neighborhood

def fcost(S):
    print("computing fcost")
    #au début on peut faire juste moins les gigaflops et appres on pourra rajouter des paramètres
    x= S[0]
    y= S[1]
    z= S[2]

    #obtention du nom du fichier automatiquement pour pas réécrire l'avx utilisé
    filepath = glob.glob('/usr/users/cpust75/cpust75_15/Proj-Intel/Appli-iso3dfd/bin/*.exe')[0]
    filename = splitext(basename(filepath))[0]

    #ecriture de la commande pour recupérer les Gflops
    command= "/usr/users/cpust75/cpust75_15/Proj-Intel/Appli-iso3dfd/bin/"+filename+".exe "+str(x)+" "+str(y)+" "+str(z)+"4 100 32 32 32"
      
    sub = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    subprocess_return = sub.stdout.read()
    
    print(subprocess_return[len(subprocess_return)-14:len(subprocess_return)-8])
    print(subprocess_return)
    cost= -float(subprocess_return[len(subprocess_return)-14:len(subprocess_return)-8])

    return (cost)

def fcostlist(liste):
    E=[]
    for element in liste:
        E.append(fcost(element))
    E= np.array(E)
    return E


def neighborhood(S,r):
    print("computing neighborhood...")
    x= S[0]
    y= S[1]
    z= S[2]
    L=[]
    X= [x-i*16 for i in range(1,r+1)]+[x+i*16 for i in range(1,r+1)]
    Y= [y-i for i in range(1,r+1)]+[y+i for i in range(1,r+1)]
    Z= [z-i for i in range(1,r+1)]+[z+i for i in range(1,r+1)]
    #à changer car approche trop naive
    for i in X:
        for j in Y:
             for k in Z:
                L.append([i,j,k])
    #somelists= [X,Y,Z]
    #L= itertools.product(*somelists) #fait le produit cartésien normalement

    return L

def greedy(Sbest,Ebest,L,kmax,NewbetterS,r,Me,NbP):
    Listebest=[]
    k=0
    n = len(L)
    q = n//NbP
    rest = n%NbP
    j=Me*q
    if j==n-rest:
        liste_p= [j+i for i in range(q+rest)]#i+j
    else : 
        liste_p= [j+i for i in range(q)]

    while (k<kmax and NewbetterS):
        print(len(L))
        print("iteration n° {}, NbP = {}, q = {}, Me={}  ".format(k,NbP,q,Me) + 40*"==")
        TabE= np.zeros(len(liste_p))
        print(liste_p)
        for i in liste_p :
           print("i=",i) 
           print("i-j = ", i-j)
           print("L[i]= ", L[i])
           TabE[i-j]= fcost(L[i])
        jp=0
        for i in range(len(liste_p)):
            if TabE[i]<TabE[jp]:
                jp=i
        E= TabE[jp]
        print("j = {} and jp = {}".format(j,jp))
        jbest=j+jp
        print(E)
        Mi=[E,Me]
        a,b = comm.allreduce(Mi, MPI.MINLOC)
        print("new E = {}, rank = {}, Ebest={}".format(a,b,Ebest))
        E= a
        rank = b
        if E<Ebest:
            jbest=comm.bcast(jbest,root= rank)
            S = L[jbest]
            Sbest=S
            Ebest=E
            Listebest.append(Ebest)
            L= neighborhood(Sbest,r)
            print("New Ebest found")

        else: 
            NewbetterS= True
        k=k+1
    print(Listebest)
    return(Sbest,Ebest)

L= neighborhood(Sbest,r)
Ebest= fcost(Sbest)
Smin= greedy(Sbest,Ebest,L,kmax,NewbetterS,r,Me,NbP)

print(Smin)

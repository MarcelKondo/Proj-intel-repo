import os
import sys
import time
import numpy as np
import math
import random as rd
from mpi4py import MPI

import general_config as GC
from numpy.core.arrayprint import SubArrayFormat

from server_content.automated_compiling_tabu import find_number, define_exec_param, define_copiler_settings, Cost

comm = MPI.COMM_WORLD
NbP = comm.Get_size()
Me = comm.Get_rank()


#def get_neighbourhood(S):
#    LNgbh = []
  
#    for param in param_space.keys():
 #       if param == 'simdType':
 #           p_idx = param_space[param].index(S[param])
 #           S1 = S.copy()
 #           if p_idx + 1 < len(param_space[param]):
 #               S1[param] = param_space[param][p_idx + 1]
 #               LNgbh.append(S1)
            
 #           S2 = S.copy()
  #          if p_idx - 1 >= 0:
 #               S2[param] = param_space[param][p_idx - 1]
  #              LNgbh.append(S2)
  #      else:
  ##          for k in range(1,3):
  #             Skp = S.copy()
  #              Skm = S.copy()
#
 #               Skp[param] += k*param_space[param][2]
 #               if Skp[param] < param_space[param][1]:
#                    LNgbh.append(Skp)
#
 #               Skm[param] -= k*param_space[param][2]
 #               if Skm[param] > 0:
   #                 LNgbh.append(Skm)
  #  return LNgbh

def ComputeC(S,Sb,eb,listparam):
                 c=[]
                 for param in listparam:
                    Sloc=Sb.copy()
                    Sloc[param]=S[param]
                    cost= Cost(Sloc)#Sb ou on ne change qu'un param comme il l'est dans S
                    c.append(cost-eb)
                 return c
            
def fcost(S,penalties, Sb, eb,listparam,lba):
  
  fcost=Cost(S)
  w=1 #pour l'instant
  c= ComputeC(S,Sb,eb,listparam)
    
    #suppose que l'ordre des penalties est le meme que celui de listeparam
  for i in range(len(listparam)):
    prox= abs(S[listparam[i]]-Sb[listparam[i]])
    fcost+=penalties[i]*prox*c[i]*lba
  return(fcost)

def fifo_add(Sb, L_tabu, tabu_size):
    if len(L_tabu)==tabu_size:
        L_tabu.pop(0)
    L_tabu.append(Sb)
    return L_tabu

def find_best(LNgbh,L_tabu, NbP, Me,penalties,c,listparam,lba,Sb,eb) : #?? parall??liser
    e = 0
    S = None
    n = len(LNgbh)
    q = n//NbP
    rest = n%NbP
    j = Me*q
    if Me<rest:
      liste_p = [LNgbh[i+j] for i in range(q)]
      liste_p.append(LNgbh[j+Me])
    else:
      liste_p = [LNgbh[i+j] for i in range(q)]
    for Sp in liste_p:
        if Sp not in L_tabu:       
            ep = fcost(Sp,penalties, Sb, eb,listparam,lba)
            #print('ep',ep)
            if ep > e :
                S = Sp
                e = ep
    #print("me e",Me,e)
    mi= [e,Me]
    e,rank = comm.allreduce(mi,op=MPI.MAXLOC)
    #print("BROADCAST")
    S= comm.bcast(S, root=rank)
    return S, e

def parallel_tabu_greedy(S0,IterMax,tabu_size,NbP, Me,penalties,c, listparam,lba):  
   
    Sb = S0
    #print("so",S0)
    eb= Cost(Sb)
    eb = fcost(Sb,penalties, Sb, eb,listparam,lba)
    print("fcost dans parall greedy")
    iter = 0
    NewBetterS = True

    S = Sb
    e = eb
    LNgbh = GC.nghbrhd_other(S)
    L_tabu = [Sb]
    
    while iter < IterMax and NewBetterS:
        S,e = find_best(LNgbh,L_tabu, NbP, Me,penalties,c,listparam,lba,Sb,eb) 
        if e > eb:
            #print("Eb GLOBAL TROUV??")
            Sb = S
            eb = e
            L_tabu = fifo_add(Sb, L_tabu, tabu_size)
            LNgbh = GC.nghbrhd_other(Sb)
            #print(len(LNgbh))
        else:
            NewBetterS = False
        iter += 1
        print(20*"=","NEW ITERATION",20*"=")
    print("[G] END")
    
    return eb,Sb,iter
                 
  
def ChoosePenaltyFeatures(p,c):
  s = len(p)*[0]
  a= sum(p)
  for i in range(len(p)):
    s[i] = c[i]/(p[i]+1)
  index_max = s.index(min(s))
  print(20*"%",index_max,20*"%")
  p[index_max]+=1
  if a==sum(p):
    p[rd.randrange(0, len(p))]+=1
  return p

def Guided(S0,IterMax,tabu_size,NbP, Me,IterMaxG,lba):
                 listparam= ['n1','n1','n1','tblock1','tblock2','tblock3']
                 penalties=[0]*len(listparam)
                 c= [0]*len(listparam)
                 #lba=0.35#?? tester
                 
                 #premier local search
                 eb,Sb,iterb= parallel_tabu_greedy(S0,IterMax,tabu_size,NbP, Me,penalties,c,listparam,lba)
                 
                 NewBetterSG= True
                 iterG=0
                 S = Sb
                 e = eb
                 #fcost = 
                 while iterG < IterMaxG and NewBetterSG:
                      
                      c=ComputeC(S,Sb,eb,listparam) #Cost(Sb)## pb ici S pas assign?? avant ComputeC(S,Sb) un fcost n'est pas calcul?? avan
                      penalties= ChoosePenaltyFeatures(penalties,c)
                      
                      e,S,iter = parallel_tabu_greedy(Sb,IterMax,tabu_size,NbP, Me,penalties,c,listparam,lba)
                      costS = Cost(S)
                      
                      if costS>eb:
                        Sb = S
                        eb = costS
                        iterb=iter #pas sur que ca ait vraiment un sens
                      else:
                        NewBetterSG = False
                      iterG += 1
                 
                 penaltie= [0]*len(listparam)
                 eb,Sb,iterb= parallel_tabu_greedy(Sb,IterMax,tabu_size,NbP, Me,penaltie,c,listparam,lba)
                 
                 return eb,Sb,iterb,penalties,c,iterG

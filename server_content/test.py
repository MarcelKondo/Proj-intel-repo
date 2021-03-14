

S = [240, 240, 240, 3, 100, 32, 32, 32]
lmax = [280, 280, 280, 8, 100, 80, 80, 80]
lmin = [32, 32, 32, 1, 100, 16, 16, 16]
param_indices = [0,1,2]


def Neighborhood(S, param_indices):                                                
    LNgbh = []
  
    for i in param_indices:
        S1 = S.copy()
        if(i == 0 or i == 5 or i == 6 or i == 7):
            S1[i] += 16
        else:
            S1[i] += 4
        if S1[i] <= lmax[i]:
            LNgbh.append(S1)
            print(LNgbh)
        
        S2 = S.copy()
        if(i == 0 or i == 5 or i == 6 or i == 7):
            S2[i] -= 16
        else:
            S2[i] -= 4
        if S2[i] >= lmin[i]:
            LNgbh.append(S2)
            print(LNgbh)
    
    return LNgbh

print(Neighborhood(S,param_indices))
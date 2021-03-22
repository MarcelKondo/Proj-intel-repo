def ChoosePenaltyFeatures(p,c):
  s = len(p)*[0]
  for i in range(len(p)):
    s[i] = c[i]/(p[i]+1)
  index_max = s.index(max(s))
  p[index_max]+=1
  return p


def ComputeC(S,Sb,keys):
  for key in keys: 
  c = ATERMINER
  
    
    
    
    

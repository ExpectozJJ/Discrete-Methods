import numpy as np 
from scipy.optimize import linprog

# linprog uses min cTx and constraints Ax <= b with x >= 0 
# Convert any linear program into the above form for input.

# MAS711 AY1617 Qn 3
# Dual LP (LP given in the qn)
c = [-1,-1,-5]
A = [[1,1,2],[-1,0,-3],[2,1,7]]
b = [3,-2,5]

linprog(c,A,b)

# Primal LP 
linprog(b,-np.transpose(A),c)
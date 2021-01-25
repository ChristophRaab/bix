import numpy as np
from bix.utils.gsmo_solver import GSMO
import pandas as pd
import os

if 0:
    pwd = os.path.dirname(os.path.abspath(__file__))
    test_data_file = os.path.join(pwd, "demo_snip_iris.csv")
    data = pd.read_csv(test_data_file, delimiter=',',header=None)
    D = data.to_numpy()
    A = D[:,np.r_[0:10]]
    b = D[:,np.r_[0]].transpose()
    b=b.reshape((10,))[:]
    C = D[:,np.r_[11:13]].transpose()
    d = np.array([1,2])
    # print(f'A: {A}')
    # print(f'b: {b}')
    # print(f'C: {C}')
    # print(f'd: {d}')
    #C = None
    #d = 0

## example 1
if 0:
    A = np.array([[1,-1],[-1,2]])
    b = np.array([-2,-6])
    C = None
    d = 0

## example 2
if 1:
    A = np.array([[3 , -2 ,  0], [-2,    6,   1,],[ 0,1, 2]])
    b = np.array([2,6,3])
    C = np.array([[-3.2,-2.8,-1.6],[-6.4,-5.6,-3.2]])
    d = np.array([2,4])
    print(f'{A}')
    print(f'{b}')
    print(f'{C}')
    print(f'{d}')

# - this (below) is only inital solvable lsqr in range -10:10 but is finally not the optimal solution
if 0:
    A = np.array([[1,0],[0,1]])
    b = np.array([1,-1])
    C = np.array([[1,1]])
    d = np.array([7]) # original 7 - the solution for 4.5 and 7 is correct in the way that the constraints are fulfilled - is the lsqr to strict? - no the other solver solved the inequality instead of equality constraints
# Matlab quadprog will give [2.5 4.5] -- all solution are valid but the cvx is optimal (costs)

H = np.array([[5,8,1],[8,13,1],[1,1,2]])
f = np.array([-16,-25,4])
C2 = np.array([[0,0,0],[0,0,0]])
d2 = np.array([0,0])

#A = np.array([[1, 0], [0, 1]], dtype=float)
#b = np.array([1, -1], dtype=float).reshape((2,))
#lb = -1
#ub = 1
#oSMO = GSMO(A=A, b=b, bounds=(lb, ub), step_size=0.1)

#oSMO = GSMO(A,b)  # bounds eigentlich pro variable nicht global
#oSMO = GSMO(A,b,C,d,(0,1e10))  # bounds eigentlich pro variable nicht global
#oSMO = GSMO(A,b,C,d,(-1,10))  # bounds eigentlich pro variable nicht global
oSMO = GSMO(A,b,C,d,(-0.3,0.3))  # bounds eigentlich pro variable nicht global
#oSMO = GSMO(A,b,None,0,(-1,1))  # bounds eigentlich pro variable nicht global -- case simon without constraints (to check)
#oSMO = GSMO(H,f,C2,d2,(0,10))
oSMO.solve()

print(oSMO.x)

if  A  is not None:
    Qsmo = oSMO.x.transpose().dot(A.dot(oSMO.x)) + b.transpose().dot(oSMO.x)
    print(f'{Qsmo}')

if   C  is not None:
    print(f'{C.dot(oSMO.x)}')
from ctypes import *
import numbers
import numpy as np
import os


class LibqpState(Structure):
    _fields_ = [("nIter", c_uint32), ("QP", c_double), ("QD", c_double), ("exitFlag", c_int8)]


"""
libqp.h: Library for Quadratic Programming optimization.
 The library provides a c binding to following solver:
 Solver for QP task with box constraints and a single linear equality constraint. 
    libqp_gsmo.c: implementation of the Generalized SMO algorithm.
 
    DESCRIPTION
    The library provides function which solves the following instance of
    a convex Quadratic Programming task:
     
    min QP(x) := 0.5*x'*H*x + f'*x  
    x                                      
     
    s.t.    a'*x = b 
            LB[i] <= x[i] <= UB[i]   for all i=1..n
     
    A precision of the found solution is controlled by the input argument
    TolKKT which defines tightness of the relaxed Karush-Kuhn-Tucker 
    stopping conditions.
     
    INPUT ARGUMENTS
    H [double n x n] matrix
    diag_H [double n x 1] vector containing values on the diagonal of H.
    f [double n x 1] vector.
    a [double n x 1] Vector which must not contain zero entries.
    b [double 1 x 1] Scalar.
    LB [double n x 1] Lower bound; -inf is allowed.
    UB [double n x 1] Upper bound; inf is allowed.
    x [double n x 1] solution vector; must be feasible.
    n [uint32_t 1 x 1] dimension of H.
    MaxIter [uint32_t 1 x 1] max number of iterations.
    TolKKT [double 1 x 1] Tightness of KKT stopping conditions.
    print_state  print function; if == NULL it is not called.
     
    RETURN VALUE
    structure [libqp_state_T]
    .QP [1x1] Primal objective value.
    .exitflag [1 x 1] Indicates which stopping condition was used:
          -1  ... not enough memory
           0  ... Maximal number of iterations reached: nIter >= MaxIter.
           4  ... Relaxed KKT conditions satisfied. 
    .nIter [1x1] Number of iterations.
     
    REFERENCE
    S.S. Keerthi, E.G. Gilbert. Convergence of a generalized SMO algorithm 
    for SVM classier design. Technical Report CD-00-01, Control Division, 
    Dept. of Mechanical and Production Engineering, National University 
    of Singapore, 2000. 
    http://citeseer.ist.psu.edu/keerthi00convergence.html  
     
     
    Copyright (C) 2006-2008 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
    Center for Machine Perception, CTU FEL Prague
     
    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public 
    License as published by the Free Software Foundation; 
    Version 3, 29 June 2007
"""


class LibQP:

    def __init__(self, H, f, a, b, LB, UB, x, maxIter, tolKKT, printfunc=None):

        if (not isinstance(H, np.ndarray)) or (not H.shape[0] == H.shape[1]):
            raise ValueError("H must be a quadratic numpy ndarray!")

        if (not isinstance(f, np.ndarray)) or (not len(f) == H.shape[0]):
            raise ValueError("f must be a n x 1 numpy ndarray!")

        if (not isinstance(a, np.ndarray)) or (not len(a) == H.shape[0]) or (not np.count_nonzero(a) == a.shape[0]):
            raise ValueError("a must be a n x 1 numpy ndarray which must not contain zero entries!")

        if not isinstance(b, numbers.Number):
            raise ValueError("b must be a scalar!")

        if (not isinstance(LB, np.ndarray)) or (not len(LB) == H.shape[0]):
            raise ValueError("LB must be a n x 1 numpy ndarray!")

        if (not isinstance(UB, np.ndarray)) or (not len(UB) == H.shape[0]):
            raise ValueError("UB must be a n x 1 numpy ndarray!")

        if (not isinstance(x, np.ndarray)) or (not len(x) == H.shape[0]):
            raise ValueError("x must be a n x 1 numpy ndarray!")

        if not isinstance(maxIter, int):
            raise ValueError("maxIter must be a int")

        if not isinstance(tolKKT, float):
            raise ValueError("tolKKT must be a float!")

        # c code needs a column wise matrix
        self.H = H.transpose()
        # necessary for data transfer
        self.H = np.ascontiguousarray(self.H, dtype=H.dtype)
        self.diagH = np.array(np.diag(H))
        self.f = f
        self.a = a
        self.b = b
        self.LB = LB
        self.UB = UB
        self.x = x
        self.maxIter = maxIter
        self.tolKKT = tolKKT

        self.dll = cdll.LoadLibrary(os.path.abspath("libqp.so"))

        def printState(strct):
            print(strct.nIter)
            print(strct.QP)
            print(strct.QD)
            print(strct.exitFlag)

        if printfunc is None:
            self.printfunc = printState
        else:
            self.printfunc = printfunc

        self.__get_column_proto = CFUNCTYPE(c_void_p, c_uint32)
        self.__print_state_proto = CFUNCTYPE(None, LibqpState)
        self.__printState = self.__print_state_proto(self.printfunc)

        self.solveSMO = self.__wrap_function("libqp_gsmo_solver", LibqpState,
                                             [np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C"),
                                              POINTER(c_double), POINTER(c_double),
                                              POINTER(c_double), c_double, POINTER(c_double), POINTER(c_double),
                                              POINTER(c_double), c_uint32, c_uint32, c_double,
                                              self.__print_state_proto])

    def __wrap_function(self, name, restype, argtypes):
        func = self.dll.__getattr__(name)
        func.restype = restype
        func.argtypes = argtypes
        return func

    def solve(self):
        columns = (self.H.__array_interface__['data'][0] + np.arange(self.H.shape[0]) * self.H.strides[0]).astype(
            np.uintp)

        return self.solveSMO(columns, self.diagH.ctypes.data_as(POINTER(c_double)),
                             self.f.ctypes.data_as(POINTER(c_double)),
                             self.a.ctypes.data_as(POINTER(c_double)), self.b,
                             self.LB.ctypes.data_as(POINTER(c_double)), self.UB.ctypes.data_as(POINTER(c_double)),
                             self.x.ctypes.data_as(POINTER(c_double)), self.H.shape[0], self.maxIter, self.tolKKT,
                             self.__printState)

from ctypes import *
import numbers
import numpy as np
import os


class LibqpState(Structure):
    _fields_ = [("nIter", c_uint32), ("QP", c_double), ("QD", c_double), ("exitFlag", c_int8)]


class LibQP:

    def __init__(self, H, f, a, b, LB, UB, x, maxIter, tolKKT, printfunc=None):

        if not isinstance(H, np.ndarray) or not H.shape[0] == H.shape[1]:
            raise ValueError("H must be a quadratic numpy ndarray!")

        if not isinstance(f, np.ndarray) or not len(f) == H.shape[0]:
            raise ValueError("f must be a n x 1 numpy ndarray!")

        if not isinstance(a, np.ndarray) or not len(a) == H.shape[0]:
            raise ValueError("a must be a n x 1 numpy ndarray!")

        if not isinstance(b, numbers.Number):
            raise ValueError("b must be a scalar!")

        if not isinstance(LB, np.ndarray) or not len(LB) == H.shape[0]:
            raise ValueError("LB must be a n x 1 numpy ndarray!")

        if not isinstance(UB, np.ndarray) or not len(UB) == H.shape[0]:
            raise ValueError("UB must be a n x 1 numpy ndarray!")

        if not isinstance(x, np.ndarray) or not len(x) == H.shape[0]:
            raise ValueError("x must be a n x 1 numpy ndarray!")

        if not isinstance(maxIter, int):
            raise ValueError("maxIter must be a int")

        if not isinstance(tolKKT, float):
            raise ValueError("tolKKT must be a float!")

        # c code needs a column wise matrix
        H = H.transpose()
        self.H = np.ascontiguousarray(H, dtype=H.dtype)
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
                                              POINTER(c_double), POINTER(c_double), \
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

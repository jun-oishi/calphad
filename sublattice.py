#! /bin/python3

import numpy as np
from typing import overload
from scipy import optimize

R = 8.314

def ln(x):
    if isinstance(x, np.ndarray):
        _x = x.copy()
        _x[x<=0] = 1
        return np.log(_x)
    else:
        return np.log(x) if x>0 else 0


class B2Binary:
    Z = 8

    def __init__(self, GAA: float = 0, GAB: float = 0, GBB: float = 0):
        self.GAA = GAA
        self.GAB = GAB
        self.GBB = GBB
        self.L1A = np.array([0])
        self.L1B = np.array([0])
        self.L2A = np.array([0])
        self.L2B = np.array([0])
        self.recL = np.array([0])

    @overload
    def Gibbs(self, y1A: float, y2A: float, T: float) -> float: ...

    @overload
    def Gibbs(self, y1A: np.ndarray, y2A: np.ndarray, T: float, mesh:bool) -> np.ndarray: ...

    @overload
    def Gibbs(self, y1A: np.ndarray, y2A: np.ndarray, T: float) -> np.ndarray: ...

    def Gibbs(self, y1A, y2A, T, mesh=False):

        # assert (
        #     (isinstance(y1A, float) and y1A>=0 and y1A<=1)
        #     or (isinstance(y1A, np.ndarray) and np.all(y1A>=0) and np.all(y1A<=1))
        # )
        # assert (
        #     (isinstance(y2A, float) and y2A>=0 and y2A<=1)
        #     or (isinstance(y2A, np.ndarray) and np.all(y2A>=0) and np.all(y2A<=1))
        # )

        if isinstance(y1A, np.ndarray) and isinstance(y2A, np.ndarray):
            if mesh:
                mesh = np.meshgrid(y1A, y2A, indexing="ij")
                y1A, y2A = mesh
            else:
                assert y1A.shape == y2A.shape


        y1B = 1 - y1A
        y2B = 1 - y2A

        h0 = ( y1A * y2A * self.GAA + y1A * y2B * self.GAB
               + y1B * y2A * self.GAB + y1B * y2B * self.GBB )
        st = 0.5 * R * T * (
                y1A * ln(y1A) + y1B * ln(y1B) + y2A * ln(y2A) + y2B * ln(y2B)
        )
        gex = 0

        assert len(self.L1A) == len(self.L1B)
        assert len(self.L2A) == len(self.L2B)
        for i in range(self.L1A.size):
            gex += (
                y1A * y1B * (y1A - y1B) ** i * (y2A * self.L1A[i] + y2B * self.L1B[i])
            )

        for i in range(self.L2A.size):
            gex += (
                y1A
                * y1B
                * (y1A - y1B) ** i
                * (y2A**2 * self.L2A[i] + y2B**2 * self.L2B[i])
            )

        for i in range(self.recL.size):
            gex += (
                y1A
                * y1B
                * y2A
                * y2B
                * 0.5
                * self.recL[i]
                * ((y1A - y1B) ** i + (y2A - y2B) ** i)
            )

        return h0 + st + gex

    def opt_g(self, xA, T) -> optimize.OptimizeResult:
        """compute Gibbs energy for a given total composition xA and temperature T.
        optimize sublattice fractions y1A and y2A to minimize Gibbs energy.
        returns the minimum Gibbs energy and the corresponding y1A fraction.
        """
        f = lambda y1A: self.Gibbs(y1A, 2 * xA - y1A, T)
        min_y1A = max(0, 2*xA-1)
        max_y1A = min(1, 2*xA)
        return optimize.minimize_scalar(f, bounds=(min_y1A, max_y1A), method="bounded")

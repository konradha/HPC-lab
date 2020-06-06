"""
Collection of linear algebra operations and CG solver
"""
from mpi4py import MPI
import numpy as np
from . import data
from . import operators

def hpc_dot(x, y):
    """Computes the inner product of x and y"""
    # ... implement ...
    # following the C implementation
    res = np.zeros(1, dtype=np.float64)
    res_global = np.zeros(1, dtype=np.float64)
    xn = x.inner.flatten()
    yn = y.inner.flatten()
    res = np.array([np.dot(xn, yn)])
    x.domain.comm_cart.Allreduce([res, MPI.DOUBLE], [res_global, MPI.DOUBLE], MPI.SUM)
    return res_global[0]

def hpc_norm2(x):
    """Computes the 2-norm of x"""
    # ... implement ...
    res = np.zeros(1, dtype=np.float64)
    res_global = np.zeros(1, dtype=np.float64)
    xn = x.inner.flatten()
    res[0] = np.linalg.norm(xn) ** 2
    x.domain.comm_cart.Allreduce([res, MPI.DOUBLE], [res_global, MPI.DOUBLE], MPI.SUM)
    return np.sqrt(res_global[0])

class hpc_cg:
    """Conjugate gradient solver class: solve the linear system A x = b"""
    def __init__(self, domain):
        self._Ap = data.Field(domain)
        self._r  = data.Field(domain)
        self._p  = data.Field(domain)

        self._xold  = data.Field(domain)
        self._v  = data.Field(domain)
        self._Fxold  = data.Field(domain)
        self._Fx  = data.Field(domain)
        self._v  = data.Field(domain)

    def solve(self, A, b, x, tol, maxiter):
        """Solve the linear system A x = b"""
        # initialize
        A(x, self._Ap)
        self._r.inner[...] = b.inner[...] - self._Ap.inner[...]
        self._p.inner[...] = self._r.inner[...]
        delta_kp = hpc_dot(self._r, self._r)
 
        # iterate
        converged = False
        for k in range(0, maxiter):
            delta_k = delta_kp
            if delta_k < tol**2:
                converged = True
                break
            A(self._p, self._Ap)
            alpha = delta_k/hpc_dot(self._p, self._Ap)
            x.inner[...] += alpha*self._p.inner[...]
            self._r.inner[...] -= alpha*self._Ap.inner[...]
            delta_kp = hpc_dot(self._r, self._r)
            self._p.inner[...] = ( self._r.inner[...]
                                  + delta_kp/delta_k*self._p.inner[...] )

        return converged, k + 1


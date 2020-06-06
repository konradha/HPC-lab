# ------------------------------------------------------------------------
# INSPIRED BY DOCS ON BITBUCKET
#
#  Poisson problem. This problem is modeled by the partial
#  differential equation
#
#          -Laplacian(u) = 20,  0 < x,y < 1,
#
#  with boundary conditions
#
#           u = 0  for  x = 0, x = 1, y = 0, y = 1
#
#  A finite difference approximation with the usual 7-point stencil
#  is used to discretize the boundary value problem to obtain a
#  nonlinear system of equations. The problem is solved in a 2D
#  rectangular domain, using distributed arrays (DAs) to partition
#  the parallel grid.
#
# ------------------------------------------------------------------------

try: range = xrange
except: pass

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

class Poisson2D(object):

    def __init__(self, da):
        assert da.getDim() == 2
        self.da = da
        self.localX  = da.createLocalVec()

    def formRHS(self, B):
        b = self.da.getVecArray(B)
        mx, my = self.da.getSizes()
        hx, hy = [1.0/m for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                b[i, j] = 20*hx*hy

    def mult(self, mat, X, Y):
        #
        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        # reference to local array
        y = self.da.getVecArray(Y)
        #
        mx, my = self.da.getSizes()
        hx, hy = [1.0/m for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                u = x[i, j] # center
                u_e = u_w = u_n = u_s = 0
                if i > 0:    u_w = x[i-1, j] # west
                if i < mx-1: u_e = x[i+1, j] # east
                if j > 0:    u_s = x[i, j-1] # south
                if j < ny-1: u_n = x[i, j+1] # north
                u_xx = (-u_e + 2*u - u_w)*hy/hx
                u_yy = (-u_n + 2*u - u_s)*hx/hy
                y[i, j] = u_xx + u_yy

OptDB = PETSc.Options()
comm = PETSc.COMM_WORLD
t1 = - PETSc.Log.getTime()
n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', n)
ny = OptDB.getInt('ny', n)

da = PETSc.DMDA().create([nx, ny], stencil_width=1)
pde = Poisson2D(da)

x = da.createGlobalVec()
b = da.createGlobalVec()
# A = da.createMat('python')
# A = PETSc.Mat()
# A.setType(PETSc.Mat.Type.SCATTER)

A = PETSc.Mat().createPython(
    [x.getSizes(), b.getSizes()], comm=da.comm)
A.setPythonContext(pde)
A.setUp()

ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType('lcd') #best: cg close: fcg, qcg, lcd
pc = ksp.getPC()
pc.setType('none')
#pc.setType('icc')
ksp.setFromOptions()

pde.formRHS(b)
t1 += PETSc.Log.getTime()

t2 = -PETSc.Log.getTime()
ksp.solve(b, x)
t2 += PETSc.Log.getTime()

ksp.view()
PETSc.Sys.Print(f"{comm.getSize()} {ksp.getIterationNumber()} {nx} {t1:.4f} {t2:.4f}")

u = da.createNaturalVec()
da.globalToNatural(x, u)
#x = x+5
#u = u+5
# collect solution on rank 0 to plot it
rank = comm.getRank()
scatter, U = PETSc.Scatter.toZero(u)
scatter.scatter(u, U, False, PETSc.Scatter.Mode.FORWARD)
if rank == 0:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    mx, my = da.sizes
    sol = U[...].reshape(da.sizes, order='f')
    X, Y = np.mgrid[0:1:1j*mx,0:1:1j*my]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, sol, cmap=mpl.cm.coolwarm)
    fig.colorbar(surf)
    #plt.plot(X.ravel(), Y.ravel())
    plt.savefig(f"sol_{nx}.png", dpi=400)
comm.barrier()

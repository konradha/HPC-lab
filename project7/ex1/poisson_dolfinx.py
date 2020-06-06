import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.plotting
import ufl
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh, solve
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from ufl import ds, dx, grad, inner

mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0., 0., 0.]), np.array([1., 1., 0.])], [128, 128],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)

V = FunctionSpace(mesh, ("Lagrange", 1))
u0 = Function(V)
u0.vector.set(0.)
facets = locate_entities_boundary(mesh, 1,
        lambda x: np.logical_or(x[0] < np.finfo(float).eps,
                                x[0] > 1.-np.finfo(float).eps)
        )
bc = DirichletBC(u0, local_dofs_topological(V, 1, facets))
u = ufl.TrialFunction(V)
v = ufl.TrialFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = ufl.mathfunctions.ConstantValue(20.)

a = inner(grad(u), grad(v)) * dx
L = -inner(f, v) * dx

u = Function(V)
solve(a == L, u, bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
dolfinx.plotting.plot(u)
plt.show()

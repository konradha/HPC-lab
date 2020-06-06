from dolfin import *
from dolfin_adjoint import *
import moola
mesh = UnitSquareMesh(64, 64)
V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "CG", 1)
m = Function(W, name='Control')
u = Function(V, name='State')
v = TestFunction(V)
# Define weak problem
F = (inner(grad(u), grad(v)) - m*v)*dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)
# Define regularisation parameter
alpha = Constant(1e-6)
# Define desired temperature profile
x = SpatialCoordinate(mesh)
d = (1/(2*pi**2) + 2*alpha*pi**2)*sin(pi*x[0])*sin(pi*x[1])
control = SteadyParameter(m)
J = Functional((0.5*inner(u-d, u-d))*dx + alpha/2*m**2*dx)
rf = ReducedFunctional(J, control)
# Set up moola problem and solve optimisation
problem = rf.moola_problem()
m_moola = moola.DolfinPrimalVector(m, inner_product="L2")
solver = moola.BFGS(problem, m_moola)
sol = solver.solve()

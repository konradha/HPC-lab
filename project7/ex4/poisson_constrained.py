import sys
import ipopt
import numpy as np
import matplotlib.pyplot as plt


class example(object):
    def __init__(self, alpha, h, N):
        self.alpha = alpha
        self.h = h
        self.N = N
        self.x, self.y = np.linspace(0,1,N+2), np.linspace(0,1,N+2)
        self.grid = np.meshgrid(self.x, self.y) 

        self.jac = np.zeros(shape=(N**2,N**2))
        """
        for i in range(N**2):
           jac.col(i)=flatten(matrix with I(i),J(i)) 
        """
        def buildJacVec(i, N):
            I = i // N
            J = i % N
            mat = np.zeros(shape=(N,N))
            for u in range(N):
                for v in range(N):
                    if I == u and v == J:
                        mat[u,v] = 4
                    if I == u+1 and v == J:
                        mat[u,v] = -1
                    if I == u-1 and v == J:
                        mat[u,v] = -1
                    if I == u and v-1 == J:
                        mat[u,v] = -1
                    if I == u and v+1 == J:
                        mat[u,v] = -1
            return mat.flatten()


        for i in range(N**2):
            self.jac[:,i] = buildJacVec(i, N)
    
    def objective(self, x):
        # convention NxN + 4*N
        # [y, u]     
        def yd(x1, x2):
            return 3. + 5*x1*(x1-1)*x2*(x2-1)
        def ud(x1, x2):
            0.
        N = self.N
        r1 = 0.
        r2 = 0.
        for i in range(N**2):
            I = i // N 
            J = i % N
            r1 += (x[i] - yd(I*h, J*h)) ** 2
        for i in range(N**2, N**2 + 4*N):
            r2 += x[i] ** 2
        return self.h**2 * .5 * r1 + self.alpha * self.h * .5 * r2

    def gradient(self, x):
        N = self.N
        h = self.h
        def yd(x1, x2):
            return 3. + 5*x1*(x1-1)*x2*(x2-1)
        def dof_mapper(i):
            return i//N, i%N
        g1 = 0.
        g2 = 0.
        for i in range(N**2):
            I, J = dof_mapper(i)
            g1 += x[i] - yd(I*h, J*h)
        g1 *= self.h
        for i in range(N**2, N**2 + 4*N):
            g2 += x[i]
        return g1 + self.alpha*self.h*g2

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #

        # 1) 
        N = self.N
        h = self.h
        G = np.zeros((N,N))
        y = np.zeros((N,N))
        u = np.zeros((4*N))
        d = -20.
        #u = np.zeros((N+2,N+2))
        for i in range(N**2):
            I = i // N 
            J = i % N
            y[I,J] = x[i]
        for i in range(N**2, N**2 + 4*N):
            u[i] = x[i]

        for I in range(1, N):
            for J in range(1, N):
                G[I, J] = 4*y[I,J] - y[I-1,J] - y[I+1,J] - y[I,J-1] - y[I,J+1] + h**2*d

        for J in range(1, N):
            G[0, J] = 4*y[0,J] - y[1,J] - u[J] - y[0,J+1] - y[0,J-1] + h**2 * d

        for I in range(1, N):
            G[I,N-1] = 4*y[I,N-1] - y[I-1,N-1] - y[I+1,N-1] - u[N+I] - y[I,N-1] + h**2 * d

        for J in reversed(range(1,N)):
            G[N-1, J] = 4*y[N-1,J] - y[1,J] - u[3*N-J-1] - y[N-1,J+1] - y[N-1,J-1] + h**2 * d

        for I in reversed(range(1, N)):
            G[I,0] = 4*y[I,0] - y[I-1,0] - y[I+1,0] - y[I,0] - u[4*N-I-1] + h**2 * d

        G[0,0]   = 4*y[0,0] - y[1, 0] - u[0] - y[0,1] - u[4*N-1] + h**2 * d
        G[0,N-1] = 4*y[0,N-1] - y[1,N-1] - u[N-1] - u[N] - y[0,N-2] + h**2 * d

        G[N-1,N-1] = 4*y[N-1,N-1] - u[2*N] - y[N-2,N-1] - u[2*N-1] - y[N-1,N-2] + h**2 * d
        G[N-1,0] = 4*y[N-1,0] - u[3*N-1] - y[N-2,0] - y[N-1,1] - u[3*N] + h**2 *d

        return np.array(G.flatten()) # x[0:(N+2)**2], x[(N+2)**2:((N+2)**2 + 4*N)]))

    def jacobian(self, x):
        return self.jac

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
N = 10

x0 = np.zeros(((N)**2 + 4*N))

lb = np.concatenate((-2.e19 * np.ones((N**2)), np.zeros((4*N))))
ub = np.concatenate((3.5 * np.ones((N**2)), 10*np.ones((4*N))))

cl = np.zeros((N)**2)
cu = np.zeros((N)**2)

nlp = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=example(0.01, 1./(N+1), N),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )
#
# Set solver options
#
nlp.addOption('tol', 1e-7)

#
# Scale the problem (Just for demonstration purposes)
#
"""
nlp.setProblemScaling(
    obj_scaling=2,
    x_scaling=[1, 1, 1, 1]
    )
nlp.addOption('nlp_scaling_method', 'user-scaling')
"""


x, info = nlp.solve(x0)

print("Solution of the primal variables: x=%s\n" % repr(x))
print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))
print("Objective=%s\n" % repr(info['obj_val']))

"""
plt.plot(x)

plt.show()
"""



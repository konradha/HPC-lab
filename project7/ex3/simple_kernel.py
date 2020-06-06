import scipy
import numpy as np

N = 10000

A = np.random.rand(N,N)
B = np.random.rand(N,N)

C = np.dot(A, B)

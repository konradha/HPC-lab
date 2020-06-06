from mpi4py import MPI
import numpy as np


# mostly adapted from the unit tests of mpi4py github repo
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ndims = 2
dims = [0]*ndims
dims = MPI.Compute_dims(size, dims)
periods = [True] * len(dims)
topo = comm.Create_cart(dims, periods=periods)
coords = topo.Get_coords(rank)
nw, ne = topo.Shift(0, 1)
ns, nn = topo.Shift(1, 1)


topo.send(rank, dest=ns)
rcv_n = topo.recv(source=nn)

topo.send(rank, dest=nn)
rcv_s = topo.recv(source=ns)

topo.send(rank, dest=ne)
rcv_w = topo.recv(source=nw)

topo.send(rank, dest=nw)
rcv_e = topo.recv(source=ne)

for i in range(size):
    # output to analyze the resulting topology for correctness
    if i == rank:
        print(f"rank {rank}")
        print(
        f"""
________________________________________________
|                {rcv_n}                             |
|                                               |
| {rcv_w}            {coords}            {rcv_e}     |
|                                               |
|                {rcv_s}                             |
-------------------------------------------------
         """)
        
"""
print(f"{rank} has n: {rcv_n}")
print(f"{rank} has w: {rcv_w}")
print(f"{rank} has s: {rcv_s}")
print(f"{rank} has e: {rcv_e}")
"""

from mpi4py import MPI
import numpy as np


# mostly adapted from the unit tests of mpi4py github repo
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ndims = 2
dims = [0]*ndims
dims = MPI.Compute_dims(size, dims)
periods = [True] * ndims
topo = comm.Create_cart(dims, periods=periods)
coords = topo.Get_coords(rank)
nw, ne = topo.Shift(0, 1)
ns, nn = topo.Shift(1, 1)
localgrid = np.array([[rank]*8]*8, dtype='i')
req = []

if rank == 1:
    print("BEFORE\n ---------------------")
    print(localgrid)

bN = localgrid[:,-1].flatten()
bS = localgrid[:,0].flatten()
bE = localgrid[-1,:].flatten()
bW = localgrid[0,:].flatten()
bufN = np.zeros_like(bN)
bufS = np.zeros_like(bS)
bufE = np.zeros_like(bE)
bufW = np.zeros_like(bW)

for i in [[bN, nn, bufN, ns], [bS, ns, bufS, nn], [bE, ne, bufE, nw], [bW, nw, bufW, ne]]:
    req.append(topo.Isend(
        [
            i[0], MPI.INT
        ],
        dest=i[1])
        )
    req.append(topo.Irecv(
        [
            i[2], MPI.INT
        ],
        source=i[1])
        )

req.append(topo.Isend([bN, MPI.INT], dest=nn))
req.append(topo.Irecv([bufN, MPI.INT], source=nn))

stat = [MPI.Status() for _ in req]

MPI.Request.Waitall(req, stat)
localgrid[:,-1] = bufN
localgrid[:,0] = bufS
localgrid[-1,:] = bufE
localgrid[0,:] = bufW

if rank == 1:
    print("AFTER\n ---------------------")
    print(localgrid)

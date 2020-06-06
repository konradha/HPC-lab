import mpi4py
from mpi4py import MPI
from random import randint
import numpy as np
from sys import argv


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if argv[1] is None:
        # trying out a collective uppercase operation
        num = np.array([randint(0, 100)], dtype='i')
        print(f"rank {rank} has {num[0]}")
        rcv = np.zeros((1,), dtype='i')
        # arbitrarily assign rank 0 to be the master rank
        comm.Reduce([num, 1, MPI.INT], [rcv, 1, MPI.INT], op=MPI.SUM, root=0)
        if rank == 0:
            print(f"\nsum is {rcv[0]}")

    elif str(argv[1]) == 'pickle':
        sums = 0
        sendple = rank
        recvple = 0
        for i in range(size):
            comm.send(sendple, dest=(rank-1+size) % size)
            recvple = comm.recv(source=(rank+1)%size)
            sums += recvple
            sendple = recvple
        print(f"rank {rank} has {sums} and should be {int(np.sum(np.linspace(0,size-1,size)))}")

    elif str(argv[1]) == 'buf':
        # going with ring topology now
        sums = np.zeros(1, dtype='i')
        sendbuf = np.array([rank], dtype='i')
        recvbuf = np.array([0],   dtype='i')
        for i in range(size):
            comm.Send([sendbuf, MPI.INT], (size + rank-1) % size)
            comm.Recv([recvbuf, MPI.INT], (rank+1)%size)
            sums += recvbuf[0]
            sendbuf[0] = recvbuf[0]
        print(f"rank {rank} has {sums[0]} and should be {int(np.sum(np.linspace(0,size-1,size)))}")

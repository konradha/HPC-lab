from mandelbrot_task import mandelbrot, mandelbrot_patch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI # MPI_Init and MPI_Finalize automatically called
import numpy as np
import sys
import time

# some parameters
MANAGER = 0       # rank of manager
TAG_TASK      = 1 # task       message tag
TAG_TASK_DONE = 2 # tasks done message tag
TAG_DONE      = 3 # done       message tag

def manager(comm, tasks):
    """
    The manager.

    Parameters
    ----------
    comm : mpi4py.MPI communicator
        MPI communicator
    tasks : list of objects with a do_task() method perfroming the task
        List of tasks to accomplish

    Returns
    -------
    ... ToDo ...
    """


    ##### THIS IS A PICKLED VERSION
    #     ----> will have to be improved if looking for performance
    #####

    patches = [] 
    curr = 0
    size = comm.Get_size()
    status = MPI.Status()
    ranktimes_recv = [0.]*size
    ranktimes_work = [0.]*size
    # kick off the workers
    for rank in range(1,size):
        comm.send([tasks[curr]], dest=rank)
        curr += 1
    # main work loop
    while curr < len(tasks):
        recvpatch = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        patches.append(recvpatch[0])
        t_recv = recvpatch[1] 
        t_work = recvpatch[2]
        curr_rank = status.source
        ranktimes_recv[curr_rank] += t_recv 
        ranktimes_work[curr_rank] += t_work
        comm.send([-1], dest=curr_rank)
        comm.send([tasks[curr]], dest=curr_rank)
        curr += 1
    # collect the last tasks and kill all workers
    for rank in range(1, size):    
        recvpatch = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        patches.append(recvpatch[0])
        curr_rank = status.source
        ranktimes_recv[curr_rank] += recvpatch[2]
        ranktimes_work[curr_rank] += recvpatch[2]
        comm.send([100], dest=curr_rank) 

    return patches, ranktimes_recv, ranktimes_work


def worker(comm):

    """
    The worker.

    Parameters
    ----------
    comm : mpi4py.MPI communicator
        MPI communicator
    """
    rank = comm.Get_rank()
    t = -time.perf_counter()
    task = (comm.recv(source=0))[0]
    t += time.perf_counter()
    t1 = -time.perf_counter()
    task.do_work()
    t1 += time.perf_counter() 
    comm.send([task,t,t1],dest=0)
    s = -1
    while True: 
        s = int(comm.recv(source=0)[0]) 
        if int(s) ==  -1: 
            t_recv = -time.perf_counter() # communication + performance
            task = (comm.recv(source=0))[0]
            t_recv += time.perf_counter()
            t_work = -time.perf_counter()
            task.do_work()
            t_work += time.perf_counter()
            
            comm.send([task,t_recv,t_work],dest=0)
        else:
            return
         

def readcmdline(rank):
    """
    Read command line arguments

    Parameters
    ----------
    rank : int
        Rank of calling MPI process

    Returns
    -------
    nx : int
        number of gridpoints in x-direction
    ny : int
        number of gridpoints in y-direction
    ntasks : int
        number of tasks
    """
    # report usage
    if len(sys.argv) != 4:
        if rank == MANAGER:
            print("Usage: manager_worker nx ny ntasks")
            print("  nx     number of gridpoints in x-direction")
            print("  ny     number of gridpoints in y-direction")
            print("  ntasks number of tasks")
        sys.exit()

    # read nx, ny, ntasks
    nx = int(sys.argv[1])
    if nx < 1:
        sys.exit("nx must be a positive integer")
    ny = int(sys.argv[2])
    if ny < 1:
        sys.exit("ny must be a positive integer")
    ntasks = int(sys.argv[3])
    if ntasks < 1:
        sys.exit("ntasks must be a positive integer")

    return nx, ny, ntasks



def perform(comm, nx, ny, ntasks):
    size    = comm.Get_size()
    my_rank = comm.Get_rank()
    timespent = -MPI.Wtime()#-time.perf_counter()
    x_min= -2.
    x_max  = +1.
    y_min  = -1.5
    y_max  = +1.5
    times = []
    if my_rank == 0:
        M = mandelbrot(x_min, x_max, nx, y_min, y_max, ny, ntasks)
        tasks = M.get_tasks()
        #print("starting manager...")
        patches, times = manager(comm, tasks)
        m = M.combine_tasks(patches)
        #print("done combining patches")
    else:
        worker(comm)
       
    if my_rank == 0:
        # there's is virtually no difference between the two methods to measure time
        timespent += MPI.Wtime()#time.perf_counter()
        times[0] = timespent  
        maxes = max(times)
        avg = [1-abs((i-maxes)/maxes) for i in times] 
        print(size, nx, ny, *times)
        #print(*avg)


if __name__ == "__main__":
    # get COMMON WORLD communicator, size & rank
    comm    = MPI.COMM_WORLD
    size    = comm.Get_size()
    my_rank = comm.Get_rank()

    # report on MPI environment
    #if my_rank == MANAGER:
    #    print(f"MPI initialized with {size:5d} processes")

    # read command line arguments
    nx, ny, ntasks = readcmdline(my_rank)

    # start timer


    """
    n = 5
    if my_rank == 0:
        print(f"starting up with {nx}x{ny}, {ntasks} tasks on {size} ranks")
    for i in range(n):
        perform(comm, nx, ny, ntasks)
    """




    
    timespent = - time.perf_counter()

    # trying out ... YOUR MANAGER-WORKER IMPLEMENTATION HERE ...

    x_min = -2.
    x_max  = +1.
    y_min  = -1.5
    y_max  = +1.5
    times = []
    if my_rank == 0:
        M = mandelbrot(x_min, x_max, nx, y_min, y_max, ny, ntasks)
        tasks = M.get_tasks()
        #print("starting manager...")
        patches, times_recv, times_work = manager(comm, tasks)
        m = M.combine_tasks(patches)
        #print("done combining patches")
    else:
        worker(comm)
       
    if my_rank == 0:
        timespent += time.perf_counter()
        times_work[0] = timespent  
        times_recv[0] = timespent
        maxes_work = max(times_work)
        maxes_recv = max(times_recv)
        avg_recv = sum(times_recv)/len(times_recv)
        avg_work = sum(times_work)/len(times_work)
        imb_recv = abs(maxes_recv-avg_recv)/maxes_recv
        imb_work = abs(maxes_work-avg_work)/maxes_work


        print(*[f"{i:.3f}" if type(i)==float else f"{i}" for i in [size, nx, ny, ntasks, maxes_recv, maxes_work, 1-imb_recv, 1-imb_work]])
        #print(*avg)
        
        
        #plt.imshow(m.T, cmap="gray", extent=[x_min, x_max, y_min, y_max])
        #plt.savefig("mandelbrot.png", dpi=500) 
        #print(f"Run took {timespent:f} seconds")
        #for i in range(size):
        #    if i == MANAGER:
        #        continue
        #    #print(f"Process {i:5d} has done {TasksDoneByWorker[i]:10d} tasks")
        #print("Done.")
    

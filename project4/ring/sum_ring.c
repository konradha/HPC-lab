/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS-USI Summer School.    *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS/USI take no responsibility for the use of the enclosed  *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Parallel sum using a ping-pong                      *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>


int main (int argc, char *argv[])
{
    int my_rank, size;
    int snd_buf, rcv_buf;
    int right, left;
    int sum, i;

    MPI_Status  status;
    MPI_Request request;


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);


    right = (my_rank + 1) % size; // right
    left  = (my_rank - 1 + size) % size ; // left needs to be adapted for cyclicality

    /* Implement ring addition code
     * do not use if (rank == 0) .. else ..
     * every rank sends initialy its rank number to a neighbor, and then sends what
     * it receives from that neighbor, this is done n times with n = number of processes
     * all ranks will obtain the sum.
     */

    sum = 0;
    snd_buf = my_rank;
    for(int i=0;i<size;++i)
    {
        MPI_Send(&snd_buf, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
        MPI_Recv(&rcv_buf, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);
        snd_buf = rcv_buf;
        sum += rcv_buf;
    }


    /*
    MPI_Send(&snd_buf, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
    MPI_Recv(&rcv_buf, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);
    snd_buf = rcv_buf;
    sum += rcv_buf;

    MPI_Send(&snd_buf, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
    MPI_Recv(&rcv_buf, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);
    snd_buf = rcv_buf;
    sum += rcv_buf;

    MPI_Send(&snd_buf, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
    MPI_Recv(&rcv_buf, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);
    sum += rcv_buf;
    */


    printf("sum for rank %i is: %i\n", my_rank, sum);


    MPI_Finalize();
    return 0;
}

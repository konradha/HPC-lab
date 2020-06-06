/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS-USI Summer School     *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS/USI take no responsibility for the use of the enclosed  *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Exchange ghost cell in 2 directions using a topology*
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/* Use only 16 processes for this exercise
 * Send the ghost cell in two directions: left<->right and top<->bottom
 * ranks are connected in a cyclic manner, for instance, rank 0 and 12 are connected
 *
 * process decomposition on 4*4 grid
 *
 * |-----------|
 * | 0| 1| 2| 3|
 * |-----------|
 * | 4| 5| 6| 7|
 * |-----------|
 * | 8| 9|10|11|
 * |-----------|
 * |12|13|14|15|
 * |-----------|
 *
 * Each process works on a 6*6 (SUBDOMAIN) block of data
 * the D corresponds to data, g corresponds to "ghost cells"
 * xggggggggggx
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * xggggggggggx
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define true 1
#define false 0

#define SUBDOMAIN 6
#define DOMAINSIZE (SUBDOMAIN+2)

int main(int argc, char *argv[])
{
    int rank, size, i, j, dims[2], periods[2], rank_top, rank_bottom, rank_left, rank_right;
    double data[DOMAINSIZE*DOMAINSIZE];
    MPI_Request request;
    MPI_Status status;
    MPI_Comm comm_cart;
    MPI_Datatype data_ghost;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size!=16) {
        printf("please run this with 16 processors\n");
        MPI_Finalize();
        exit(1);
    }

    // initialize the domain
    for (i=0; i<DOMAINSIZE*DOMAINSIZE; i++) {
        data[i]=rank;
    }

	int m = SUBDOMAIN;
	int n = DOMAINSIZE;

    // set the dimensions of the processor grid and periodic boundaries in both dimensions
    dims[0] = 4;
    dims[1] = 4;
	// use booleans since even though they're ints, they're flags (y/n) for periodicity
	// built macro since only CPP compilers cast type
    periods[0] = true;
    periods[1] = true;

    // create a Cartesian communicator (4*4) with periodic boundaries (we do not allow
    // the reordering of ranks) and use it to find your neighboring
    // ranks in all dimensions in a cyclic manner.
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);

    // find your top/bottom/left/right neighbor using the new communicator, see MPI_Cart_shift()
    // rank_top, rank_bottom
    // rank_left, rank_right
	MPI_Cart_shift(comm_cart, 0, 1, &rank_top, &rank_bottom);
	MPI_Cart_shift(comm_cart, 1, 1, &rank_left, &rank_right);

    // create derived datatype data_ghost, create a datatype for sending the column, see MPI_Type_vector() and MPI_Type_commit()
    // data_ghost
	MPI_Type_vector(m, 1, n, MPI_DOUBLE, &data_ghost);
	MPI_Type_commit(&data_ghost);

    //  ghost cell exchange with the neighbouring cells in all directions
  	/*
	int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
					int dest, int sendtag,
					void *recvbuf, int recvcount, MPI_Datatype recvtype,
					int source, int recvtag,
					MPI_Comm comm, MPI_Status *status)
	*/

	
	// ALIGNED DATA: use usual MPI_DATATYPE MPI_DOUBLE

	//  to the top
	MPI_Sendrecv(&data[2 - 1 + (2-1) * n], m, MPI_DOUBLE, rank_top, 0b01, &data[2 - 1 + (n-1) * n], m, MPI_DOUBLE, rank_bottom, 0b01, comm_cart, MPI_STATUS_IGNORE);

    //  to the bottom
	MPI_Sendrecv(&data[2 - 1 + (n-1-1) * n], m, MPI_DOUBLE, rank_bottom, 0b10, &data[2 - 1 + (1-1)*n], m, MPI_DOUBLE, rank_top, 0b10, comm_cart, MPI_STATUS_IGNORE);
		
	// UNALIGNED DATA: use MPI_DATATYPE data_ghost
	// MPI deduces stride from topology definition

    //  to the left
	MPI_Sendrecv(&data[2 - 1 + (2-1) * n], 1, data_ghost, rank_left, 0b00, &data[n - 1 + (2-1)*n], 1, data_ghost, rank_right, 0b00, comm_cart, MPI_STATUS_IGNORE);
    
    //  to the right
	MPI_Sendrecv(&data[n - 1 - 1 + (2-1) * n], 1, data_ghost, rank_right, 0b11, &data[1-1 + (2-1)*n], 1, data_ghost, rank_left, 0b11, comm_cart, MPI_STATUS_IGNORE);
	

    if (rank==9) {
        printf("data of rank 9 after communication\n");
        for (j=0; j<DOMAINSIZE; j++) {
            for (i=0; i<DOMAINSIZE; i++) {
                printf("%.1f ", data[i+j*DOMAINSIZE]);
            }
            printf("\n");
        }
    }

    MPI_Type_free(&data_ghost);
    MPI_Comm_free(&comm_cart);
    MPI_Finalize();

    return 0;
}

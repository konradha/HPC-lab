/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS-USI Summer School.    *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS/USI take no responsibility for the use of the enclosed  *
 * teaching material.                                           *
 *                                                              *
 * Purpose: : Parallel matrix-vector multiplication and the     *
 *            and power method                                  *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

#include "hpc-power.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


// A*x=y \in R(nxn)
void matVec(double* A, double* x, double* y, int N)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // vector lies on 0
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double* local = (double*)calloc(N/size, sizeof(double));
    for(int i=0; i<N/size; i++)
        for(int j=0; j<N  ; j++)
            local[i] += A[i*N + j] * x[j];
    // find sol
    MPI_Gather(local, N/size, MPI_DOUBLE, y, N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(local);
}

double norm(double* x, int N)
{
    double res=.0;
    // should probably unroll here
    for(int i=0;i<N;++i)
        res += x[i]*x[i];
    return sqrt(res);

}

double powerMethod(double* A, int n, int N) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status; MPI_Request request;
    double *x = (double *) calloc(N, sizeof(double));
    if (rank == 0)
        for (int i = 0; i < N; i++)
            x[i] = (double) (rand() % 10) / 100;

    for (int i = 0; i < n; ++i) {
        if (rank == 0) {
            double l2 = norm(x, N);
            for (int k = 0; k < N; ++k)
                x[k] /= l2;
        }
        //MPI_Wait(&request, &status);
        double *iter = (double *) calloc(N, sizeof(double));
        matVec(A, x, iter, N);
        if (rank == 0) {
            for (int k = 0; k < N; ++k)
                x[k] = iter[k];
        }
    }
    return norm(x, N);
}

void generateMatrix(double* A, int N)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for(int i = 0; i < N/size; ++i)
        for(int j = 0; j < N; ++j)
            A[i*N + j] = 1.;
}

void generateNew(double* A, int N, double s)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for(int i = 0; i < N/size; ++i)
        for(int j = 0; j < N; ++j) {
            if(i == j)
                A[i * N + j] = 2.*s;
            if (abs(i - j) == 1)
                A[i * N + j] = -s*.5;
        }
}


void print(double* A, int N, int isMat)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0) {
        if(isMat) {
            for (int i = 0; i < N/size; ++i) {
                for (int j = 0; j < N; ++j) {
                    printf("%.3f ", A[i*N + j]);
                }
                printf("\n");
            }
        } else {
           for(int i=0;i<N/size;++i)
               printf("%.3f \n", A[i]);
        }
    }
}


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
    int N = 1<<12;
    //if(my_rank == 0)
    //    printf("N mod size: %i\n", N%size);

    double* A = (double*)calloc(N*N/size, sizeof(double)) ;//(double*)malloc(N * N/size);

    //generateMatrix(A, N);
    generateNew(A, N, 277.);
    //double* A = hpc_generateMatrix(N, N/size * my_rank, N/size);

    //print(A, N, 1);
    //double *y;
    //double *x;

    //if(my_rank == 0) {
    //    //y = (double*)calloc(N, sizeof(double));
    //    x = (double *) calloc(N, sizeof(double));
    //    for(int i=0;i<N;++i)
    //        x[i] = 1.;

    //}
    //y = (double*)calloc(N, sizeof(double));
    //matVec(A, x, y, N);
    //print(y, N, 0);

    double time = -hpc_timer();
    double res = powerMethod(A, 1000, N);
    time += hpc_timer();
    if(my_rank == 0) {
        printf("%i, %i, %.4f\n", size, N, time);
    }




    /* This subproject is about to write a parallel program to
       multiply a matrix A by a vector x, and to use this routine in
       an implementation of the power method to find the absolute
       value of the largest eigenvalue of the matrix. Your code will
       call routines that we supply to generate matrices, record
       timings, and validate the answer.
    */

    free(A);
    if(my_rank == 0) {
        //free(y);
        //free(x);
    }
    MPI_Finalize();
    return 0;
}


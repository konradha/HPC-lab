#include <omp.h>
#include <iostream>
#include "walltime.h"
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#define NUM_ITERATIONS 100

// Example benchmarks
// 0.008s ~0.8MB
//define N 100000
// 0.1s ~8MB
// #define N 1000000
// 1.1s ~80MB
// #define N 10000000
// 13s ~800MB 
#define N 100000000
// 127s 16GB 
//#define N 1000000000
#define EPSILON 0.1


using namespace std;

int main(int argc, char** argv)
{
	//int N = int(strtol(argv[1], NULL, 10));//1<<int(strtol(argv[1], NULL, 10));

    int myId, numTdreads;
    double time_serial, time_start=0.0;
    long double dotProduct;
    double *a,*b;

    // Allocate memory for the vectors as 1-D arrays
    a = new double[N];
    b = new double[N];
  
    // Initialize the vectors with some values
    for(int i=0; i<N; i++) {
	a[i] = i;
	b[i] = i/10.0;
    }

    // getting the strong feeling that the compiler optimizes this loop away
    volatile long double alpha = 0;
    // serial execution
    // Note that we do extra iterations to reduce relative timing overhead
    time_start = wall_time();
    for( int iterations=0; iterations<NUM_ITERATIONS; iterations++) {
	alpha=0.0;
	for( int i=0; i<N; i ++) {
	    alpha += a[i] * b[i];
    	}
    }
    time_serial = wall_time() - time_start;
    //cout << "Serial execution time = " << time_serial << " sec" << endl;
  
    long double alpha_parallel = 0;
    double time_red=0;
    double time_critical=0;

   	time_red = -wall_time();
    for( int iterations=0; iterations<NUM_ITERATIONS; iterations++) {
	alpha_parallel=0.0;
#pragma omp parallel for reduction(+:alpha_parallel)
	for( int i=0; i<N; i ++) {
	    alpha_parallel += a[i] * b[i];
    	}
    }
	time_red += wall_time();

	double alpha_critical =0.;
	time_critical = -wall_time();
    for( int iterations=0; iterations<NUM_ITERATIONS; iterations++) {
	alpha_critical=0.0;
	double local_var = 0.;
#pragma omp parallel for
	for( int i=0; i<N; i ++) {
	    local_var = a[i] * b[i];
#pragma omp critical
		alpha_critical += local_var;
    	}
    }
	time_critical += wall_time();
	

#pragma omp parallel
#pragma omp single
	std::cout << omp_get_num_threads() << "," << N << "," << time_serial << "," << time_red << "," << time_critical << "\n";


	/*
	cout << "par / crit " << alpha_parallel << " " << alpha_critical << "\n";

    if( (fabs(alpha_parallel - alpha)/fabs(alpha_parallel)) > EPSILON) {
	cout << "parallel reduction: " << alpha_parallel << " serial :" << alpha << "\n";
	cerr << "Alpha not yet implemented correctly!\n";
	exit(1);
    }
    cout << "Parallel dot product = " << alpha_parallel
	 << " time using reduction method = " << time_red
	 << " sec, time using critical method " << time_critical
	 << " sec" << endl;
	 */
  
    // De-allocate memory
    delete [] a;
    delete [] b;

    return 0;
}

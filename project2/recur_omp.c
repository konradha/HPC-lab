# include <stdlib.h>
# include <math.h>
# include "walltime.h"
#include <omp.h>

int main ( int argc, char *argv[] ) {
    int N = 2000000000;
    double up = 1.00000001 ;
    double Sn = 1.00000001;
    int n;
    /* allocate memory for the recursion */
    double* opt = (double*) malloc ((N+1)* sizeof(double));

    if (opt == NULL)  die ("failed to allocate problem size");

    double time_start = wall_time();	
	double offset;
	long m, t;
#pragma omp parallel private(offset, t) shared(m) reduction(*: Sn)
	{
		t = omp_get_thread_num();
		m = omp_get_num_threads();
		offset = pow(up, t * N / m);
		long i;
		for(i=0;i<N/m;++i)
		{
			opt[i + t*N/m] = Sn * offset;
			Sn *= up;	
		}
		/*
 * #pragma omp barrier
 *
 * #pragma omp single 
 * 		Sn = pow(Sn, m);
 * 				*/
	}
	
	/*
 * #pragma omp parallel for  shared(Sn) lastprivate(tmp) reduction(*: Sn)  
 *     for (n = 0; n <= N; ++n) {
 *     	tmp = Sn;
 *     		opt[n] = tmp;
 *     			Sn *= up;
 *     			    }
 *     			    	*/
	

	
    printf("Parallel RunTime   :  %f seconds\n", wall_time()- time_start);
    printf("Final Result Sn    :  %e \n", Sn );

    double temp = 0.0;
    for (n = 0; n <= N; ++n) {
	temp +=  opt[n] * opt[n];
    }
    printf("Result ||opt||^2_2 :  %f\n", temp/(double) N);
    printf ( "\n" );

    return 0;
}


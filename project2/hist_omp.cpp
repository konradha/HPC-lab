#include <iostream>
#include "walltime.h"
#include <stdlib.h>
#include <random>

#include <omp.h>

#define VEC_SIZE 1000000000
#define BINS 16

using namespace std;

int main()
{
    double time_start, time_end;

    // Initialize random number generator
    unsigned int seed = 123;
    float mean = BINS/2.0;
    float sigma = BINS/12.0;
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution (mean, sigma);

    // Generate random sequence
    // Note: normal distribution is on interval [-inf; inf]
    //       we want [0; BINS-1]
    int *vec = new int[VEC_SIZE];
    for(long i = 0; i < VEC_SIZE; ++i) {
        vec[i] = int(distribution(generator));
        if (vec[i] < 0)
            vec[i] = 0;
        if (vec[i] > BINS-1)
            vec[i] = BINS-1;
    }

    // Initialize histogram
    // Set all bins to zero
    long dist[BINS];
    for(int i = 0; i < BINS; ++i) {
        dist[i] = 0;
    }

    time_start = wall_time();

	
	long M = VEC_SIZE;
#pragma omp parallel shared(vec, dist)
	{	
		long local_dist[BINS]{0};
		long tid = omp_get_thread_num();
		long ts = omp_get_num_threads();
//#pragma omp parallel for shared(M, vec) lastprivate(local_dist)
		for(long i=0;i<M/ts;++i)
		{
			local_dist[vec[i + ts*tid]]++;
		}
		// leftovers for first thread
		if(ts > 1 && M % ts != 0)
		{
#pragma omp master
			{
				long m = M % ts;
				for(int i=M-m;i<M;++i)
					local_dist[vec[i]]++;
			}
		}

#pragma omp critical
		for(long i=0;i<BINS;++i)
			dist[i] += local_dist[i];
	}

    time_end = wall_time();

    // Write results
    /*
    for(int i = 0; i < BINS; ++i) {
        cout << "dist[" << i << "]=" << dist[i] << endl;
    }
*/
#pragma omp parallel
#pragma omp single
    cout << omp_get_num_threads() << "," << time_end - time_start <<  endl;

    delete [] vec;

    return 0;
}

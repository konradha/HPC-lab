/* 
COMPILER=icc

MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
*/

#include <immintrin.h>
#include <stdlib.h>
#define min(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

const char* dgemm_desc = "blocked dgemm";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */

static inline void naive_dgemm(double* __restrict__ a, double* __restrict__ b, double* __restrict__ c, int n, int s, int t, int u, int r, double* at)
{	

	for(int i=0;i<s;++i)
	{
		for(int K=0;K<u;++K)
			at[K] = a[i+K*n];
		for(int j=0;j<t;++j)
			{
				double cij = 0.;//c[i+j*n];
				int k=0;			
				// stride size is 16 now	
				int v=u-(u%16);
				__m256d cc = _mm256_set_pd(0.,0.,0.,0.);
				for(;k<v;k+=16)
				{
					__m256d a1 = _mm256_loadu_pd(at + k);
					__m256d a2 = _mm256_loadu_pd(at + k + 4);	
					__m256d a3 = _mm256_loadu_pd(at + k + 8);
					__m256d a4 = _mm256_loadu_pd(at + k + 12);	

					__m256d b1 = _mm256_loadu_pd(b + k + j*n);
					__m256d b2 = _mm256_loadu_pd(b + k + 4 + j*n);
					__m256d b3 = _mm256_loadu_pd(b + k + 8 + j*n);
					__m256d b4 = _mm256_loadu_pd(b + k + 12 + j*n);
					
					cc = _mm256_fmadd_pd(b1, a1, cc);
					cc = _mm256_fmadd_pd(a2, b2, cc);
					cc = _mm256_fmadd_pd(a3, b3, cc);
					cc = _mm256_fmadd_pd(a4, b4, cc);
				}	
				__m128d lo = _mm256_castpd256_pd128(cc);
				__m128d hi = _mm256_extractf128_pd(cc, 1);
				lo = _mm_add_pd(lo, hi);
				__m128d h64 = _mm_unpackhi_pd(lo, lo);
				cij += _mm_cvtsd_f64(_mm_add_sd(lo, h64));	
#pragma vector aligned
#pragma ivdep
				for(;k<u;++k)
				{
					cij += at[k] * b[k+j*n];
				}		
				c[i+j*n] += cij;	
			}	
	}	
}


void square_dgemm (int n, double* A, double* B, double* C)
{
	int r = 48;	
	double* a_transpose;
	int s, t, u;
#pragma omp parallel private(a_transpose, s, t, u)
	{
		a_transpose = calloc(r, sizeof(double));	
#pragma omp for collapse(2) nowait
		for(int i=0;i<n;i+=r)
			// doesn't make sense to transpose A here
			// as recommended in Project pdf
			for(int j=0;j<n;j+=r)
//#pragma omp task shared(C) depend(in: A[i]) depend(in: B[i])
				for(int k=0;k<n;k+=r)
				{
					s = min(n-i, r);	
					t = min(n-j, r);
					u = min(n-k, r);
					naive_dgemm(A+i+k*n, B+k+j*n, C+i+j*n, n, s, t, u, r, a_transpose);
				}
		free(a_transpose);
	}
}



//keeping in code for sanity checks
/*
void square_dgemm (int n, double* A, double* B, double* C)
{ 
   	for (int i = 0; i < n; ++i)
for (int j = 0; j < n; ++j) 
    {
      double cij = C[i+j*n];
      for( int k = 0; k < n; k++ )
		cij += A[i+k*n] * B[k+j*n];
      C[i+j*n] = cij;
    }
}
*/

// linear algebra subroutines
// Ben Cumming @ CSCS

#include <iostream>

#include <cmath>
#include <cstdio>

#include "linalg.h"
#include "operators.h"
#include "stats.h"
#include "data.h"

#include <immintrin.h>
#include <omp.h>

namespace linalg {

bool cg_initialized = false;
Field r;
Field Ap;
Field p;
Field Fx;
Field Fxold;
Field v;
Field xold;

using namespace operators;
using namespace stats;
using data::Field;

// initialize temporary storage fields used by the cg solver
// I do this here so that the fields are persistent between calls
// to the CG solver. This is useful if we want to avoid malloc/free calls
// on the device for the OpenACC implementation (feel free to suggest a better
// method for doing this)
void cg_init(int nx)
{
    Ap.init(nx,nx);
    r.init(nx,nx);
    p.init(nx,nx);
    Fx.init(nx,nx);
    Fxold.init(nx,nx);
    v.init(nx,nx);
    xold.init(nx,nx);

    cg_initialized = true;
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
////////////////////////////////////////////////////////////////////////////////

// computes the inner product of x and y
// x and y are vectors on length N
double hpc_dot(Field const& x, Field const& y, const int N)
{
    double result = 0;
#pragma omp parallel for simd reduction(+: result)
#pragma vector nontemporal
    for (int i = 0; i < N; i++)
        result += x[i] * y[i];

    return result;
}

// computes the 2-norm of x
// x is a vector on length N
double hpc_norm2(Field const& x, const int N)
{
    double result = 0;
/*
 * NOT actually reliably faster - compiler uses Class structure nearly optimally
 * to feed data into the pipeline
 *
 * ----> well designed Classes make for nearly optimal code!
 * ----> well designed Classes better than premature optimization to make progress
 *
	int stride = 16;
	const int residue = N % stride;
	__m256d tmp{0., 0., 0., 0.};
	const double* addr = x.data(); 
	{
		for(int i=0;i<N-residue;i+=stride)
		{
			tmp = _mm256_fmadd_pd(_mm256_loadu_pd(addr + i), _mm256_loadu_pd(addr + i), tmp);
			tmp = _mm256_fmadd_pd(_mm256_loadu_pd(addr + i + 4), _mm256_loadu_pd(addr + i + 4), tmp);
			tmp = _mm256_fmadd_pd(_mm256_loadu_pd(addr + i + 8), _mm256_loadu_pd(addr + i + 8), tmp);
			tmp = _mm256_fmadd_pd(_mm256_loadu_pd(addr + i + 12), _mm256_loadu_pd(addr + i + 12), tmp);
		}
		__m128d lo = _mm256_castpd256_pd128(tmp);
		// extract upper bits
		__m128d hi = _mm256_extractf128_pd(tmp, 1);
		lo = _mm_add_pd(lo, hi);
		// cast to format we can use & add
		__m128d h64 = _mm_unpackhi_pd(lo, lo);
		result += _mm_cvtsd_f64(_mm_add_sd(lo, h64));
	}

	for(int i=N-residue;i<N;++i)
		result += x[i] * x[i];
*/

#pragma omp parallel for simd reduction(+: result)	
#pragma vector nontemporal
	for(int i=0;i<N;++i)
		result += x[i] * x[i];
    return sqrt(result);

}

// sets entries in a vector to value
// x is a vector on length N
// value is a scalar
void hpc_fill(Field& x, const double value, const int N)
{
#pragma omp parallel for simd collapse(1)
#pragma vector nontemporal
    for(int i=0;i<N;++i)
		x[i] = value;
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 vector-vector operations
////////////////////////////////////////////////////////////////////////////////

// computes y := alpha*x + y
// x and y are vectors on length N
// alpha is a scalar
void hpc_axpy(Field& y, const double alpha, Field const& x, const int N)
{
#pragma omp parallel for simd collapse(1)
#pragma vector nontemporal
    for(int i=0;i<N;++i)
		y[i] = alpha*x[i] + y[i];
}

// computes y = x + alpha*(l-r)
// y, x, l and r are vectors of length N
// alpha is a scalar
void hpc_add_scaled_diff(Field& y, Field const& x, const double alpha,
    Field const& l, Field const& r, const int N)
{
#pragma omp parallel for simd collapse(1)
#pragma vector nontemporal
    for(int i=0;i<N;++i)
		y[i] = x[i] + alpha * (l[i] - r[i]); 
}

// computes y = alpha*(l-r)
// y, l and r are vectors of length N
// alpha is a scalar
void hpc_scaled_diff(Field& y, const double alpha,
    Field const& l, Field const& r, const int N)
{
#pragma omp parallel for simd collapse(1)
#pragma vector nontemporal
    for(int i=0;i<N;++i)
		y[i] = alpha * (l[i] - r[i]);
}

// computes y := alpha*x
// alpha is scalar
// y and x are vectors on length n
void hpc_scale(Field& y, const double alpha, Field& x, const int N)
{
#pragma omp parallel for simd collapse(1)
#pragma vector nontemporal
    for(int i=0;i<N;++i)
		y[i] = alpha * x[i];
}

// computes linear combination of two vectors y := alpha*x + beta*z
// alpha and beta are scalar
// y, x and z are vectors on length n
void hpc_lcomb(Field& y, const double alpha, Field& x, const double beta,
    Field const& z, const int N)
{
#pragma omp parallel for simd collapse(1)
#pragma vector nontemporal
    for(int i=0;i<N;++i)
		y[i] = alpha * x[i] + beta * z[i];
}

// copy one vector into another y := x
// x and y are vectors of length N
void hpc_copy(Field& y, Field const& x, const int N)
{
#pragma omp parallel for simd collapse(1)
#pragma vector nontemporal
    for(int i=0;i<N;++i)
		y[i] = x[i];
}

// conjugate gradient solver
// solve the linear system A*x = b for x
// the matrix A is implicit in the objective function for the diffusion equation
// the value in x constitute the "first guess" at the solution
// x(N)
// ON ENTRY contains the initial guess for the solution
// ON EXIT  contains the solution
void hpc_cg(Field& x, Field const& b, const int maxiters, const double tol, bool& success)
{
    // this is the dimension of the linear system that we are to solve
    using data::options;
    int N = options.N;
    int nx = options.nx;

    if (!cg_initialized)
        cg_init(nx);

    // epslion value use for matrix-vector approximation
    double eps     = 1.e-8;
    double eps_inv = 1. / eps;

    // allocate memory for temporary storage
    hpc_fill(Fx,    0.0, N);
    hpc_fill(Fxold, 0.0, N);
    hpc_copy(xold, x, N);

    // matrix vector multiplication is approximated with
    // A*v = 1/epsilon * ( F(x+epsilon*v) - F(x) )
    //     = 1/epsilon * ( F(x+epsilon*v) - Fxold )
    // we compute Fxold at startup
    // we have to keep x so that we can compute the F(x+exps*v)
    diffusion(x, Fxold);

    // v = x + epsilon*x
    hpc_scale(v, 1.0 + eps, x, N);

    // Fx = F(v)
    diffusion(v, Fx);

    // r = b - A*x
    // where A*x = (Fx-Fxold)/eps
    hpc_add_scaled_diff(r, b, -eps_inv, Fx, Fxold, N);

    // p = r
    hpc_copy(p, r, N);

    // r_old_inner = <r,r>
    double r_old_inner = hpc_dot(r, r, N), r_new_inner = r_old_inner;

    // check for convergence
    success = false;
    if (sqrt(r_old_inner) < tol)
    {
        success = true;
        return;
    }

    int iter;
    for(iter=0; iter<maxiters; iter++) {
        // Ap = A*p
        hpc_lcomb(v, 1.0, xold, eps, p, N);
        diffusion(v, Fx);
        hpc_scaled_diff(Ap, eps_inv, Fx, Fxold, N);

        // alpha = r_old_inner / p'*Ap
        double alpha = r_old_inner / hpc_dot(p, Ap, N);

        // x += alpha*p
        hpc_axpy(x, alpha, p, N);

        // r -= alpha*Ap
        hpc_axpy(r, -alpha, Ap, N);

        // find new norm
        r_new_inner = hpc_dot(r, r, N);

        // test for convergence
        if (sqrt(r_new_inner) < tol) {
            success = true;
            break;
        }

        // p = r + r_new_inner.r_old_inner * p
        hpc_lcomb(p, 1.0, r, r_new_inner / r_old_inner, p, N);

        r_old_inner = r_new_inner;
    }
    stats::iters_cg += iter + 1;

    if (!success)
        std::cerr << "ERROR: CG failed to converge" << std::endl;
}

} // namespace linalg

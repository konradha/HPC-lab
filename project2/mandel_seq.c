#include <stdlib.h>
#include <stdio.h>

#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#include "pngwriter.h"
#include "consts.h"

#include <math.h>

#include <omp.h>


unsigned long get_time ()
{
    struct timeval tp;
    gettimeofday (&tp, NULL);
    return tp.tv_sec * 1000000 + tp.tv_usec;
}

int main (int argc, char** argv)
{
	png_data* pPng = png_create (IMAGE_WIDTH, IMAGE_HEIGHT);
	
	double x, y, x2, y2, cx, cy;
	cy = MIN_Y;
	
	double fDeltaX = (MAX_X - MIN_X) / (double) IMAGE_WIDTH;
	double fDeltaY = (MAX_Y - MIN_Y) / (double) IMAGE_HEIGHT;
	long M = IMAGE_HEIGHT;
	long N = IMAGE_WIDTH;

	long max = MAX_ITERS;
	
	long nTotalIterationsCount = 0;
	unsigned long nTimeStart = get_time ();	
	long i, j, n;

#pragma omp parallel shared(cx, pPng, max, M, N, n) reduction(+: nTotalIterationsCount) 
	for (j = 0; j < M; j++)
	{	
//#pragma omp critical
		cx = MIN_X;	
#pragma omp parallel for ordered schedule(dynamic, 1) firstprivate(cx, i, j, n) lastprivate( x, x2, y, y2)
		for (i = 0; i < N; i++)
		{	 
			n = 0;
			x = 0.;//cx;
			y = 0.;//cy;
			x2 = 0.;//x * x;
			y2 = 0.;//y * y;
			// compute the orbit z, f(z), f^2(z), f^3(z), ...
			// count the iterations until the orbit leaves the circle |z|=2.
			// stop if the number of iterations exceeds the bound MAX_ITERS.
			while(x2 + y2 < 4. && n < MAX_ITERS)
			{	
				y = 2*x*y + cy;
				x = x2 - y2 + cx;
	
				x2 = x*x;
				y2 = y*y;
				n += 1;
				nTotalIterationsCount += 1;
			}	
			//int c = ((long) n * 255) / MAX_ITERS;	
#pragma omp ordered
			{
				if(x2 + y2 < 4) png_plot(pPng, i, j, 255, 255, 255);
				else png_plot (pPng, i, j, 0, 0, 0);
			}
			

			


                        // <<<<<<<< CODE IS NISSING
                        // n indicates if the point belongs to the mandelbrot set
			// plot the number of iterations at point (i, j)
			//int c = ((long) n * 255) / MAX_ITERS;
			//png_plot (pPng, i, j, x2, y2, c);

//#pragma omp critical
			cx += fDeltaX;
		}
//#pragma omp critical
		cy += fDeltaY;
	}

	unsigned long nTimeEnd = get_time ();

	// print benchmark data
	printf ("Total time:                 %g millisconds\n", (nTimeEnd - nTimeStart) / 1000.0);
	printf ("Image size:                 %ld x %ld = %ld Pixels\n",
		(long) IMAGE_WIDTH, (long) IMAGE_HEIGHT, (long) (IMAGE_WIDTH * IMAGE_HEIGHT));
	printf ("Total number of iterations: %ld\n", nTotalIterationsCount);
	printf ("Avg. time per pixel:        %g microseconds\n", (nTimeEnd - nTimeStart) / (double) (IMAGE_WIDTH * IMAGE_HEIGHT));
	printf ("Avg. time per iteration:    %g microseconds\n", (nTimeEnd - nTimeStart) / (double) nTotalIterationsCount);
	printf ("Iterations/second:          %g\n", nTotalIterationsCount / (double) (nTimeEnd - nTimeStart) * 1e6);
	// assume there are 8 floating point operations per iteration
	printf ("MFlop/s:                    %g\n", nTotalIterationsCount * 8.0 / (double) (nTimeEnd - nTimeStart));
	
	png_write (pPng, "mandel.png");
	return 0;
}

#!/bin/bash

for i in 1 2 3 4; do
	export MKL_NUM_THREADS=$i
	export NUMEXPR_NUM_THREADS=$i
	export OMP_NUM_THREADS=$i
	time python3 simple_kernel.py
done

# really poor performance (GIL?)
# parallelization does not scale well.
#real	0m40.012s
#user	0m38.927s
#sys	0m0.892s
#
#real	0m28.946s
#user	0m54.011s
#sys	0m1.208s
#
#real	0m22.804s
#user	1m3.411s
#sys	0m1.012s
#
#real	0m20.017s
#user	1m13.886s
#sys	0m1.172s

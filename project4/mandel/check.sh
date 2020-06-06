#!/bin/bash

#touch mandel_mpi.c
#make -j1
#for i in 1 2 3 4
#do
#	mpirun -np $i ./mandel_mpi
#	mv mandel.png mandel$i.png
#done

echo "---------------------------"
echo "Image comparison with idiff"

for i in 1 2 3 4
do
	for j in 2 3 4
	do
		echo -n "ranks=$i vs ranks=$j: "
		idiff mandel$i.png mandel$j.png | head -n5 | tail -n1 
   	done
done

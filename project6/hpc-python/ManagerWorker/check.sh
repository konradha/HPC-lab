#!/bin/bash

#for k in 101 1001 2001 4001
#do
#	for i in 2 3 4
#	do
#		for j in 10 25 50 75 100 150 200
#		do mpirun -np $i python3 manager_worker.py $k $k $j
#	   	done
#	done
#done


for k in 2001
do
	for i in 2 3 4
	do
		for j in 100 150 250
		do mpirun -np $i python3 manager_worker.py $k $k $j
	   	done
	done
done

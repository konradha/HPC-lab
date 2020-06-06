#!/bin/bash
for j in 16 32 64 128 256
do	
	for i in 1 2 3 4
	do
			out="$(mpirun -np $i python3 main.py $j 100 .01 | tail -n5| head -n1)"
			var1="$(echo $out | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')" 	
			echo "$i" "$j" $var1
			sudo kill -9 $(pgrep python3) &> /dev/null
	done
done

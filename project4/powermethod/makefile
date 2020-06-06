CC=mpicxx
CFLAGS=-O3 -Wall -pedantic

all: powermethod

powermethod: powermethod.c hpc-power.c 
	$(CC) $(CFLAGS) -I. $^ -o $@

clean:
	rm -f powermethod

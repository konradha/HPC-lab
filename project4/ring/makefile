CC=mpicc
CFLAGS=-O3

all: sum_ring

sum_ring: sum_ring.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f sum_ring


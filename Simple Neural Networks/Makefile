CC=gcc
CFLAGS=-I.

main: main.o simple_nn.o lin_alg.o
	$(CC) -o main main.o simple_nn.o lin_alg.o $(CFLAGS)

clean:
	rm -f *.o main

CC=gcc
CFLAGS=-I.

main: main.o simple_nn.o tests.o
	$(CC) -o main main.o simple_nn.o tests.o $(CFLAGS)

clean:
	rm -f *.o main

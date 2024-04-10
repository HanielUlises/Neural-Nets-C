CC=gcc
CFLAGS=-I.

main: main.o simple_nn.o
	$(CC) -o main main.o simple_nn.o $(CFLAGS)

clean:
	rm -f *.o main

CC = gcc
CFLAGS = -Wall -Wextra -g -fopenmp `pkg-config --cflags gtk+-3.0`
CFLAGS += -IMLP -IMNIST

LDFLAGS = `pkg-config --libs gtk+-3.0` -lm -fopenmp

OBJDIR = objects

SRC_MAIN = main.c
SRC_MNIST = mnist.c
SRC_MLP = ../MLP/MLP.c

SRCS_C = $(SRC_MAIN) $(SRC_MNIST) $(SRC_MLP)

OBJS_C = $(addprefix $(OBJDIR)/, \
	$(SRC_MAIN:.c=.o) \
	$(SRC_MNIST:.c=.o) \
	MLP.o)

EXEC = mnist

all: $(EXEC)

$(EXEC): $(OBJS_C)
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/MLP.o: ../MLP/MLP.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(EXEC)

.PHONY: clean

SHELL = /bin/bash

CC = gcc
CFLAGS = -g -std=gnu17 -O3 -march=native -fopenmp -I include

OBJ_COLORING = main.o coloring.o runner.o fast_io.o \
			   first_fit.o largest_degree_first.o smallest_degree_last.o \
			   incidence_degree.o gnn.o fast_gnn.o

OBJ_COLORING := $(addprefix bin/, $(OBJ_COLORING))

DEP = $(OBJ_COLORING)
DEP := $(sort $(DEP))

vpath %.c src
vpath %.h include

all : coloring

-include $(DEP:.o=.d)

coloring : $(OBJ_COLORING)
	$(CC) $(CFLAGS) -o $@ $^

bin/%.o : %.c
	$(CC) $(CFLAGS) -MMD -c $< -o $@ 

.PHONY : clean
clean :
	rm -f coloring
	rm -f $(DEP)
	rm -f $(DEP:.o=.d)
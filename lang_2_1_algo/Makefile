CFLAGS = --ptxas-options=-v --use_fast_math --compiler-options -Wall -g
INCLUDES =

LDFLAGS = -L/opt/cuda/lib64
LIBS = -lcublas

OBJECTS = main.o aux.o toeplitz.o householder.o Matrix.o Toeplitz.o MatrixNumerics.o
PROG = atlas

all:$(PROG)

$(PROG): $(OBJECTS)
	nvcc $(CFLAGS) $(LDFLAGS) $(LIBS) $(OBJECTS) -o $(PROG)

%.o: %.cu
	nvcc $(CFLAGS) $(INCLUDES) -c $<

clean:
	-rm -rf *.o $(PROG)

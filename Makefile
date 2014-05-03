CFLAGS = --ptxas-options=-v --use_fast_math --compiler-options -Wall
INCLUDES = 

LDFLAGS = -L cublas
LIBS = 

OBJECTS = main.o aux.o toeplitz.o householder.o
PROG = atlas

all:$(PROG)

$(PROG): $(OBJECTS)
	nvcc $(CFLAGS) $(LDFLAGS) $(LIBS) $(OBJECTS) -o $(PROG)

%.o: %.cu
	nvcc $(CFLAGS) $(INCLUDES) -c $<

clean:
	rm -rf *.o $(PROG)

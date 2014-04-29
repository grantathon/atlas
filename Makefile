main: main.cu toeplitz.cu toeplitz.h Makefile
	nvcc -o main main.cu toeplitz.cu --ptxas-options=-v --use_fast_math --compiler-options -Wall

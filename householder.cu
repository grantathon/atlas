#include "householder.cuh"
#include <stdio.h>

using namespace std;

// Reduce the matrix to a tridiagonal matrix via Householder transformations
void BlockPairReduction(float *q, float *column, float *block_pair, int dim)
{
    // TODO: Call cuBLAS functions to solve for first column
    //      of reduced matrix (b) and the next Q (q).
    
    // TODO: The function should return the Q and the first column.
}


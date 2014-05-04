#include "householder.cuh"
#include "cublas_v2.h"
#include "aux.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

// Reduce the matrix to a tridiagonal matrix via Householder transformations
int BlockPairReduction(float *q, float *column, float *block_pair, int dim)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    float *dBlockPair;
    
    cudaMalloc(&dBlockPair, (size_t)dim*dim*sizeof(*block_pair)); CUDA_CHECK;
    
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    stat = cublasSetMatrix (dim, dim, sizeof(*block_pair), block_pair, dim, dBlockPair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (dBlockPair);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    // TODO: Call cuBLAS functions to solve for first column
    //      of reduced matrix (b) and the next Q (q).
    
    //modify (handle, dBlockPair, M, N, 1, 2, 16.0f, 12.0f);
    
    stat = cublasGetMatrix (dim, dim, sizeof(*block_pair), dBlockPair, dim, block_pair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (dBlockPair);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    cudaFree (dBlockPair);
    cublasDestroy(handle);
    
    // TODO: The function should return the Q and the first column.
    
    return 0;
}


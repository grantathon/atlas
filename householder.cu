#include "householder.cuh"
#include "cublas_v2.h"
#include "aux.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

/* Compute each element of the Householder vector based on the thread index */
__global__ void ComputeHouseVec(float *v, float *a_col, float alpha, float r, int length)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(x < length && threadIdx.x != 0)
    {
        // TODO: Try with shared memory

        if(threadIdx.x == 1)
        {
            v[threadIdx.x] = (a_col[threadIdx.x] - alpha) / (2*r);
        }
        else
        {
            v[threadIdx.x] = a_col[threadIdx.x] / (2*r);
        }
    }
}

/* Compute the Q_n matrix used to tridiagonalize first row & colum of our block pair */
__global__ void ComputeQ(float *q, float *v, int dim)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(x < dim && y < dim)
    {

    }
}

/* Compute the new block pair using computed Q matrix */
__global__ void UpdateBlockPair(float *block_pair, float *q, int dim)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(x < dim && y < dim)
    {

    }
}

/* Tridiagonalize first row & colum of block pair by computing Q using Householder vector */
/*void UpdateBlockPair(float *block_pair, float *v, float *q, int dim)
{
    //ComputeQ();
    //ComputeBlockPair();
}*/

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

    /* HOUSEHOLDER TRANSFORMATION */

    float alpha = 0.0;
    float r = 0.0;
    float *v = (float *)malloc(dim*sizeof(*v));

    // Retreive first column of block pair
    printf("Before transformation:\n");
    for(int i = 0; i < dim; i++)
    {
        column[i] = block_pair[dim*i];
        printf("  column[%u] = %f\n", i, column[i]);
    }

    // Compute alpha
    for(int i = 1; i < dim; i++)
    {
        alpha += powf(column[i], 2);
    }
    if(column[1] > 0)
    {
        alpha = -sqrtf(alpha);
    }
    else
    {
        alpha = sqrtf(alpha); 
    }

    // Compute r
    r = sqrtf((powf(alpha, 2) - column[1]*alpha) / 2);

    // Allocate memory on the GPU and copy data
    float *gpuV, *gpuColumn;
    cudaMalloc(&gpuV, (size_t)dim*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&gpuColumn, (size_t)dim*sizeof(float)); CUDA_CHECK;
    cudaMemset(gpuV, 0, (size_t)dim*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(gpuColumn, column, (size_t)dim*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    // Init block and grid sizes
    dim3 block = dim3(256, 1, 1);
    dim3 grid = dim3((dim + block.x-1)/block.x, 1, 1);

    // TODO: Put all GPU operations into one kernel

    ComputeHouseVec<<<grid, block>>>(gpuV, gpuColumn, alpha, r, dim);

    // TODO: Call cuBLAS functions to solve for first column
    //      of reduced matrix (b) and the next Q (q).

    // Change block and grid sizes for matrices
    dim3 block = dim3(32, 8, 1);
    dim3 grid = dim3((dim + block.x-1)/block.x, (dim+block.y-1)/block.y, 1);

    //ComputeQ<<<grid, block>>>();
    //UpdateBlockPair<<<grid, block>>>();


    // Copy data back to CPU and free GPU memory
    cudaMemcpy(v, gpuV, (size_t)dim*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(column, gpuColumn, (size_t)dim*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(gpuV); CUDA_CHECK;
    cudaFree(gpuColumn); CUDA_CHECK;

    printf("After transformation:\n");
    for(int i = 0; i < dim; i++)
    {
        printf("  column[%u] = %f\n", i, column[i]);
    }
    for(int i = 0; i < dim; i++)
    {
        printf("  v[%u] = %f\n", i, v[i]);
    }
    
    stat = cublasGetMatrix (dim, dim, sizeof(*block_pair), dBlockPair, dim, block_pair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (dBlockPair);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    cudaFree (dBlockPair);
    cublasDestroy(handle);

    free(v);
    
    // TODO: The function should return the Q and the first column.
    
    return 0;
}


#include "householder.cuh"
#include "cublas_v2.h"
#include "aux.h"
#include "toeplitz.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

using namespace std;

/* Compute each element of the Householder vector based on the thread index */
__global__ void ComputeHouseVec(float *v, float *a_col, float alpha, float r, int n)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(x < n && threadIdx.x != 0)
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

/* Computes the Q matrix of the Householder transformation */
__global__ void ComputeQ(float *q, float *v, int dim)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(x < dim && y < dim)
    {
        const int idx = x + y*dim;

        if(x == y)  // Compute diagonal value
        {
            q[idx] = 1.0f - 2*powf(v[x], 2);
        }
        else  // Compute symmetric values
        {
            q[idx] = -2*v[x]*v[y];
        }
    }
}

/* Update the block pair using the Q matrix */
__inline__ int UpdateBlockPair(float *block_pair, float *v, float *q, int dim)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    float *gpuQ, *gpuBlockPair, *gpuTempMatrix;
    float alpha = 1.0f;
    float beta = 1.0f;
    
    // Allocate memory on the GPU
    cudaMalloc(&gpuQ, (size_t)dim*dim*sizeof(*gpuQ)); CUDA_CHECK;
    cudaMalloc(&gpuBlockPair, (size_t)dim*dim*sizeof(*gpuBlockPair)); CUDA_CHECK;
    cudaMalloc(&gpuTempMatrix, (size_t)dim*dim*sizeof(*gpuTempMatrix)); CUDA_CHECK;
    cudaMemset(gpuQ, 0, (size_t)dim*sizeof(*gpuQ)); CUDA_CHECK;

    /* Compute Q */

    dim3 block = dim3(64, 8, 1);
    dim3 grid = dim3((dim + block.x-1)/block.x, (dim + block.y-1)/block.y, 1);

    // Perform Q computation
    ComputeQ<<<grid, block>>>(gpuQ, v, dim);
    cudaMemcpy(q, gpuQ, (size_t)dim*dim*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    printf("\nNew Q:\n");
    PrintMatrix(q, dim, dim);
    printf("\n");
    
    /* Compute block pair */
    // TODO: Fix indexing issue with cuBLAS

    // Create cuBLAS handle
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        cudaFree(gpuQ);
        cudaFree(gpuBlockPair);
        cudaFree(gpuTempMatrix);
        return -1;
    }
    
    // Initialize Q matrix
    stat = cublasSetMatrix(dim, dim, sizeof(*q), q, dim, gpuQ, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(gpuQ);
        cudaFree(gpuBlockPair);
        cudaFree(gpuTempMatrix);
        cublasDestroy(handle);
        return -2;
    }
    
    // Initialize block pair matrix
    stat = cublasSetMatrix(dim, dim, sizeof(*block_pair), block_pair, dim, gpuBlockPair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(gpuQ);
        cudaFree(gpuBlockPair);
        cudaFree(gpuTempMatrix);
        cublasDestroy(handle);
        return -3;
    }

    // Perform block pair computation
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, gpuQ, dim, gpuBlockPair, dim, &beta, gpuTempMatrix, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(gpuQ);
        cudaFree(gpuBlockPair);
        cudaFree(gpuTempMatrix);
        cublasDestroy(handle);
        return -6;
    }

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, gpuTempMatrix, dim, gpuQ, dim, &beta, gpuBlockPair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(gpuQ);
        cudaFree(gpuBlockPair);
        cudaFree(gpuTempMatrix);
        cublasDestroy(handle);
        return -7;
    }

    // Retreive Q matrix
    stat = cublasGetMatrix(dim, dim, sizeof(*q), gpuQ, dim, q, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree(gpuQ);
        cudaFree(gpuBlockPair);
        cublasDestroy(handle);
        return -4;
    }

    // Retreive block pair matrix
    stat = cublasGetMatrix(dim, dim, sizeof(*block_pair), gpuBlockPair, dim, block_pair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree(gpuQ);
        cudaFree(gpuBlockPair);
        cublasDestroy(handle);
        return -5;
    }

    printf("New block pair:\n");
    PrintMatrix(block_pair, dim, dim);
    printf("\n");
    
    // Free GPU memory
    cudaFree(gpuQ);
    cudaFree(gpuBlockPair);
    cudaFree(gpuTempMatrix);
    cublasDestroy(handle);

    return 0;
}

// Reduce the matrix to a tridiagonal matrix via Householder transformations
int BlockPairReduction(float *q, float *column, float *block_pair, int dim)
{
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
    UpdateBlockPair(block_pair, gpuV, q, dim);

    // TODO: Retreive column!

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

    free(v);
    
    // TODO: The function should return the Q and the first column.
    
    return 0;
}


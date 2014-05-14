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

/* Prepare Q matrix for Householder transformation */
__global__ void ComputeQ(float *q, float *v, int dim)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(x < dim && y < dim)
    {
        if(x == y)  // Compute diagonal value
        {
            q[IDX2C(y, x, dim)] = 1.0f - 2*powf(v[x], 2);
        }
        else  // Compute symmetric values
        {
            q[IDX2C(y, x, dim)] = -2*v[x]*v[y];
        }
    }
}

/* Update the block pair using the Q matrix */
__inline__ int UpdateBlockPair(float *block_pair, float *v, float *q, int dim)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    float *cublasBlockPair, *cublasTempMatrix, *cublasQ;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate memory on the GPU
    cudaMalloc(&cublasBlockPair, (size_t)dim*dim*sizeof(*cublasBlockPair)); CUDA_CHECK;
    cudaMalloc(&cublasTempMatrix, (size_t)dim*dim*sizeof(*cublasTempMatrix)); CUDA_CHECK;
    cudaMalloc(&cublasQ, (size_t)dim*dim*sizeof(*cublasQ)); CUDA_CHECK;
   
    /* Update block pair */

    // Create cuBLAS handle
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        return -1;
    }
    
    // Initialize Q matrix
    stat = cublasSetMatrix(dim, dim, sizeof(*q), q, dim, cublasQ, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        cublasDestroy(handle);
        return -2;
    }
    
    // Initialize block pair matrix
    stat = cublasSetMatrix(dim, dim, sizeof(*block_pair), block_pair, dim, cublasBlockPair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        cublasDestroy(handle);
        return -3;
    }

    // Perform block pair computations
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, cublasQ, dim, cublasBlockPair, dim, &beta, cublasTempMatrix, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        cublasDestroy(handle);
        return -6;
    }

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, cublasTempMatrix, dim, cublasQ, dim, &beta, cublasBlockPair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        cublasDestroy(handle);
        return -7;
    }

    // Retreive block pair matrix
    stat = cublasGetMatrix(dim, dim, sizeof(*block_pair), cublasBlockPair, dim, block_pair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        cublasDestroy(handle);
        return -5;
    }

    // Free GPU memory
    cudaFree(cublasQ);
    cudaFree(cublasBlockPair);
    cudaFree(cublasTempMatrix);
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
    for(int i = 0; i < dim; i++)
    {
        column[i] = block_pair[dim*i];
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
    float *gpuV, *gpuColumn, *gpuQ;
    cudaMalloc(&gpuV, (size_t)dim*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&gpuColumn, (size_t)dim*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&gpuQ, (size_t)dim*dim*sizeof(*gpuQ)); CUDA_CHECK;
    cudaMemset(gpuV, 0, (size_t)dim*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(gpuColumn, column, (size_t)dim*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemset(gpuQ, 0, (size_t)dim*dim*sizeof(*gpuQ)); CUDA_CHECK;

    // TODO: Put all GPU operations into one kernel

    // Init block and grid sizes
    dim3 block = dim3(512, 1, 1);
    dim3 grid = dim3((dim + block.x-1)/block.x, 1, 1);

    ComputeHouseVec<<<grid, block>>>(gpuV, gpuColumn, alpha, r, dim);
    block = dim3(32, 32, 1);
    grid = dim3((dim + block.x-1)/block.x, (dim + block.y-1)/block.y, 1);

    // Perform Q computation
    cudaDeviceSynchronize();
    ComputeQ<<<grid, block>>>(gpuQ, gpuV, dim);
    cudaMemcpy(q, gpuQ, (size_t)dim*dim*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    UpdateBlockPair(block_pair, gpuV, q, dim);

    // Copy data back to CPU and free GPU memory
    cudaMemcpy(column, gpuColumn, (size_t)dim*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(gpuV); CUDA_CHECK;
    cudaFree(gpuColumn); CUDA_CHECK;
    cudaFree(gpuQ); CUDA_CHECK;

    // Retreive first column of block pair
    for(int i = 0; i < dim; i++)
    {
        column[i] = block_pair[i*dim];
    }

    free(v);
    
    return 0;
}


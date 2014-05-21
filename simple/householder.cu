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
            q[IDX2C(y, x, dim)] = 1.0f - 2.0f*powf(fabs(v[x]), 2.0);
        }
        else  // Compute symmetric values
        {
            q[IDX2C(y, x, dim)] = -2.0f*v[x]*v[y];
        }
    }
}

/* Update the block pair using the Q matrix */
int UpdateBlockPair(float *block_pair, float *v, float *q, int dim, int shift)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    float *cublasBlockPair, *cublasTempMatrix, *cublasQ, *shiftedBlockPair;
    float alpha = 1.0f;
    float beta = 0.0f;

    // Copy over the shifted block pair
    shiftedBlockPair = (float *)malloc((size_t)dim*dim*sizeof(*shiftedBlockPair));
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            shiftedBlockPair[i*dim + j] = block_pair[i*(dim + shift) + j + shift*((dim + shift) + 1)];
        }
    }
    
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
        free(shiftedBlockPair);
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
        free(shiftedBlockPair);
        return -2;
    }
    
    // Initialize block pair matrix
    stat = cublasSetMatrix(dim, dim, sizeof(*shiftedBlockPair), shiftedBlockPair, dim, cublasBlockPair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        cublasDestroy(handle);
        free(shiftedBlockPair);
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
        free(shiftedBlockPair);
        return -6;
    }

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, cublasTempMatrix, dim, cublasQ, dim, &beta, cublasBlockPair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        cublasDestroy(handle);
        free(shiftedBlockPair);
        return -7;
    }

    // Retreive block pair matrix
    stat = cublasGetMatrix(dim, dim, sizeof(*shiftedBlockPair), cublasBlockPair, dim, shiftedBlockPair, dim);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree(cublasQ);
        cudaFree(cublasBlockPair);
        cudaFree(cublasTempMatrix);
        cublasDestroy(handle);
        free(shiftedBlockPair);
        return -5;
    }

    // Free GPU memory
    cudaFree(cublasQ);
    cudaFree(cublasBlockPair);
    cudaFree(cublasTempMatrix);
    cublasDestroy(handle);

    // Copy over updated elements
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            block_pair[i*(dim + shift) + j + shift*((dim + shift) + 1)] = shiftedBlockPair[i*dim + j];
        }
    }
    
    free(shiftedBlockPair);
    return 0;
}

// Reduce the matrix to a tridiagonal matrix via Householder transformations
int BlockPairReduction(float *q, float *column, float *block_pair, int dim, int shift)
{
    /* HOUSEHOLDER TRANSFORMATION */
    cudaDeviceSynchronize();

    float alpha = 0.0;
    float r = 0.0;
    float *v = (float *)malloc((dim - shift)*sizeof(*v));

    // Retreive first column of block pair
    for(int i = 0; i < (dim - shift); i++)
    {
        column[i] = block_pair[i*dim + shift*(dim + 1)];
    }

    // Compute alpha
    for(int i = 1; i < (dim - shift); i++)
    {
        alpha += powf(fabs(column[i]), 2);
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
    cudaMalloc(&gpuV, (size_t)(dim - shift)*sizeof(*gpuV)); CUDA_CHECK;
    cudaMalloc(&gpuColumn, (size_t)(dim - shift)*sizeof(*gpuColumn)); CUDA_CHECK;
    cudaMalloc(&gpuQ, (size_t)(dim - shift)*(dim - shift)*sizeof(*gpuQ)); CUDA_CHECK;
    cudaMemset(gpuV, 0, (size_t)(dim - shift)*sizeof(*gpuV)); CUDA_CHECK;
    cudaMemcpy(gpuColumn, column, (size_t)(dim - shift)*sizeof(*gpuColumn), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemset(gpuQ, 0, (size_t)(dim - shift)*(dim - shift)*sizeof(*gpuQ)); CUDA_CHECK;

    // TODO: Put all GPU operations into one kernel

    // Init block and grid sizes
    dim3 block = dim3(512, 1, 1);
    dim3 grid = dim3(((dim - shift) + block.x-1)/block.x, 1, 1);

    ComputeHouseVec<<<grid, block>>>(gpuV, gpuColumn, alpha, r, (dim - shift));
    block = dim3(32, 32, 1);
    grid = dim3(((dim - shift) + block.x-1)/block.x, ((dim - shift) + block.y-1)/block.y, 1);

    // Perform Q computation
    cudaDeviceSynchronize();
    ComputeQ<<<grid, block>>>(gpuQ, gpuV, (dim - shift));
    cudaMemcpy(q, gpuQ, (size_t)(dim - shift)*(dim - shift)*sizeof(*gpuQ), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // UpdateBlockPair(block_pair, gpuV, q, dim);
    // UpdateBlockPair(block_pair, gpuV, q, (dim - shift));
    UpdateBlockPair(block_pair, gpuV, q, (dim - shift), shift);

    // Copy data back to CPU and free GPU memory
    cudaMemcpy(column, gpuColumn, (size_t)(dim - shift)*sizeof(*gpuColumn), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(gpuV); CUDA_CHECK;
    cudaFree(gpuColumn); CUDA_CHECK;
    cudaFree(gpuQ); CUDA_CHECK;

    // Retreive first column of block pair
    for(int i = 0; i < (dim - shift); i++)
    {
        column[i] = block_pair[i*dim + shift*(dim + 1)];
    }

    free(v);
    return 0;
}


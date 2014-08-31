#include "householder.cuh"
#include "cublas_v2.h"
#include "aux.h"
#include "toeplitz.h"

#include <iostream>
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
        float col_val = a_col[threadIdx.x];

        if(col_val != 0)  // Only work on non-zero elements
        {
            if(threadIdx.x != 1)
            {
                v[threadIdx.x] = col_val / (2*r);
            }
            else
            {
                v[threadIdx.x] = (col_val - alpha) / (2*r);
            }
        }
    }
}

/* Prepare Q matrix for Householder transformation */
__global__ void ComputeQ(float *q, float *v, int dim)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(x > 0 && x < dim && y > 0 && y < dim)
    {
        float x_val = v[x];
        float y_val = v[y];

        if(x == y)  // Compute diagonal value
        {
            q[IDX2C(y - 1, x - 1, dim - 1)] = 1.0f - 2.0f*powf(fabs(x_val), 2.0);
        }
        else  // Compute symmetric values
        {
            q[IDX2C(y - 1, x - 1, dim - 1)] = -2.0f*x_val*y_val;
        }
    }
}

/* Update the block pair using the Q matrix */
int UpdateBlockPair(float *block_pair, float *q, int dim, int shift)
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
    
    // Initialize Q matrix
    stat = cublasSetMatrix(dim, dim, sizeof(*q), q, dim, cublasQ, dim);
    
    // Initialize block pair matrix
    stat = cublasSetMatrix(dim, dim, sizeof(*shiftedBlockPair), shiftedBlockPair, dim, cublasBlockPair, dim);

    // Perform block pair computations
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, cublasQ, dim, cublasBlockPair, dim, &beta, cublasTempMatrix, dim);
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, cublasTempMatrix, dim, cublasQ, dim, &beta, cublasBlockPair, dim);

    // Retreive block pair matrix
    stat = cublasGetMatrix(dim, dim, sizeof(*shiftedBlockPair), cublasBlockPair, dim, shiftedBlockPair, dim);

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

// Reduce the matrix to a tridiagonal matrix via Bruno Lang's 2.1 SBTH algorithm
int BlockPairReduction(float *block_pair, int block_dim, int sub_block_dim, int nu)
{
    cudaDeviceSynchronize();

    int compBlocks = 0;
    int remBlocks = 0;
    float alpha = 0.0;
    float r = 0.0;
    float *v = (float *)malloc((sub_block_dim + 1)*sizeof(*v));
    float *column = (float *)malloc((block_dim - nu)*sizeof(*column));
    float *q = (float *)malloc(sub_block_dim*sub_block_dim*sizeof(*q));

    // Determine the blocks to compute and the remaining blocks s.t.
    // block_dim - nu = sub_block_dim*(compBlocks - 1) + remBlocks
    // 1 <= remBlocks <= sub_block_dim
    compBlocks = floor((block_dim - nu) / (float)sub_block_dim);
    remBlocks = block_dim - nu - sub_block_dim*(compBlocks - 1);
    if(remBlocks > sub_block_dim)
    {
        compBlocks += floor(remBlocks / sub_block_dim);
        remBlocks -= sub_block_dim*floor(remBlocks / sub_block_dim);
    }

    // Retreive first full diagonal-element column of block pair
    for(int i = 0; i < (block_dim - nu); i++)
    {
        column[i] = block_pair[(i*block_dim + nu*(block_dim + 1))];
        cout << column[i] << endl;
    }

    // Compute alpha
    for(int i = 1; i < (block_dim - nu); i++)
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

    // cout << "r = " << r << ", alpha = " << alpha << endl;

    // Allocate memory on the GPU and copy data
    float *gpuV, *gpuColumn, *gpuQ;
    cudaMalloc(&gpuV, (size_t)(sub_block_dim + 1)*sizeof(*gpuV)); CUDA_CHECK;
    cudaMalloc(&gpuColumn, (size_t)(block_dim - nu)*sizeof(*gpuColumn)); CUDA_CHECK;
    cudaMalloc(&gpuQ, (size_t)sub_block_dim*sub_block_dim*sizeof(*gpuQ)); CUDA_CHECK;

    cudaMemset(gpuV, 0, (size_t)(sub_block_dim + 1)*sizeof(*gpuV)); CUDA_CHECK;
    cudaMemcpy(gpuColumn, column, (size_t)(block_dim - nu)*sizeof(*gpuColumn), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemset(gpuQ, 0, (size_t)sub_block_dim*sub_block_dim*sizeof(*gpuQ)); CUDA_CHECK;

    // TODO: Put all GPU operations into one kernel (need CC 3.5)

    // Init block and grid sizes
    dim3 block = dim3(512, 1, 1);
    dim3 grid = dim3(((sub_block_dim + 1) + block.x-1)/block.x, 1, 1);

    ComputeHouseVec<<<grid, block>>>(gpuV, gpuColumn, alpha, r, sub_block_dim + 1);

    // cudaMemcpy(v, gpuV, (size_t)(sub_block_dim + 1)*sizeof(*gpuV), cudaMemcpyDeviceToHost); CUDA_CHECK;
    // PrintVector(v, sub_block_dim + 1);

    // Set block and grid for next computation
    block = dim3(32, 32, 1);
    grid = dim3(((sub_block_dim + 1) + block.x-1)/block.x, ((sub_block_dim + 1) + block.y-1)/block.y, 1);

    // Perform Q computation
    cudaDeviceSynchronize();
    ComputeQ<<<grid, block>>>(gpuQ, gpuV, (sub_block_dim + 1));

    cudaMemcpy(q, gpuQ, (size_t)sub_block_dim*sub_block_dim*sizeof(*gpuQ), cudaMemcpyDeviceToHost); CUDA_CHECK;
    // PrintMatrix(q, sub_block_dim, sub_block_dim);

    // Update elements of input matrix
    UpdateBlockPair(block_pair, q, sub_block_dim + 1, nu);
    // UpdateBlockPair(block_pair, gpuV, q, (block_dim - nu), nu);

    PrintMatrix(block_pair, block_dim, block_dim);
    return 0;

    // Copy data back to CPU and free GPU memory
    cudaMemcpy(column, gpuColumn, (size_t)(block_dim - nu)*sizeof(*gpuColumn), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(gpuV); CUDA_CHECK;
    cudaFree(gpuColumn); CUDA_CHECK;
    cudaFree(gpuQ); CUDA_CHECK;

    // Retreive first column of block pair
    for(int i = 0; i < (block_dim - nu); i++)
    {
        column[i] = block_pair[i*block_dim + nu*(block_dim + 1)];
    }

    free(v);
    free(column);
    free(q);

    return 0;
}


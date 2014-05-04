// ###
// ###
// ### Symmetric Banded Matrix Reduction to Tridiagonal Form via Householder Transformations
// ### 
// ###
// ### Grant Bartel, grant.bartel@tum.de
// ### Christoph Riesinger, riesinge@in.tum.de
// ###
// ### 
// ### Technical University of Munich
// ###
// ###

#include <iostream>
#include "aux.h"
#include "toeplitz.h"
#include "householder.cuh"

using namespace std;

int main(int argc, char **argv)
{
    cudaDeviceSynchronize(); CUDA_CHECK;
    
    if(argc != 3)
    {
        cout << "Enter correct amount of arguments {[dim] [diagCnt]}." << endl;
        return -1;
    }

    const int dim = atoi(argv[1]);
    const int diagCnt = atoi(argv[2]);
    float *cpuToeplitz = new float[dim*dim];
    int buildToeplitzError = 0;

    buildToeplitzError = BuildToeplitz(cpuToeplitz, dim, diagCnt);
    if(buildToeplitzError != 0)
    {
        return -2;
    }

    PrintMatrix(cpuToeplitz, dim, dim);
    
    // Allocate memory on the GPU and copy data
    float *gpuToeplitz;
    cudaMalloc(&gpuToeplitz, (size_t)dim*dim*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(gpuToeplitz, cpuToeplitz, (size_t)dim*dim*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    // Init block and grid sizes
    dim3 block = dim3(32, 8, 1);
    dim3 grid = dim3((dim+block.x-1)/block.x, (dim+block.y-1)/block.y, 1);

    // Review the grid dimensions
    cout << "grid.x=" << grid.x << " grid.y=" << grid.y << " grid.z=" << grid.z << endl;

    // Start timer
    Timer timer;
    float t = timer.get();
    timer.start();
    
    // Perform GPU computations
    
    // TODO: Test the block-pair reduction algo on one input matrix
    //      by reading the Q output and the first column of the input
    //      matrix. If this works, begin algo 3.1 of paper by distributing
    //      the work among kernels and implementing a communication
    //      method.
    
    /*for(size_t v = 0; v < vMax; v++)
    {
        // Copy toeplitz to device
        
        for(size_t b = 0; b < dim; b++)
        {
            // BlockPairReduction() on specific dimensions of toeplitz_v
            // Save Q for next call of BlockPairReduction()
            // Use column for the construction of toeplitz_v+1
        }
    }*/
    
    
    // End timer
    timer.end();  t = timer.get();  // elapsed time in seconds
    cout << "time GPU: " << t*1000<<" ms" << endl;
    
	// Free GPU memory
    cudaFree(gpuToeplitz); CUDA_CHECK;

    // Free heap memory
    delete[] cpuToeplitz;
    
    return 0;
}


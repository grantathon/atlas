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
#include <stdio.h>
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

    // Initialize computation parameters
    const int dim = atoi(argv[1]);
    const int diagCnt = atoi(argv[2]);
    float *cpuToeplitz = (float *)malloc(dim*dim*sizeof(*cpuToeplitz));
    float *cpuQ = (float *)malloc(dim*dim*sizeof(*cpuQ));
    float *cpuBlockPairCol = (float *)malloc(dim*sizeof(*cpuBlockPairCol));

    int errorCheck = BuildToeplitz(cpuToeplitz, dim, diagCnt);
    if(errorCheck != 0)
    {
        cout << "Issue when executing BuildToeplitz()" << endl;
        return -2;
    }

    printf("\nToeplitz before:\n");
    PrintMatrix(cpuToeplitz, dim, dim);

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
    
    errorCheck = BlockPairReduction(cpuQ, cpuBlockPairCol, cpuToeplitz, dim);
    if(errorCheck != 0)
    {
        cout << "Issue when executing BlockPairReduction()" << endl;
        return -3;
    }

    // End timer
    timer.end();  t = timer.get();  // elapsed time in seconds
    
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
    
    // Print results
    printf("\nToeplitz after:\n");
    PrintMatrix(cpuToeplitz, dim, dim);
    printf("\nQ:\n");
    PrintMatrix(cpuQ, dim, dim);
    printf("\nBlock pair column:\n");
    PrintVector(cpuBlockPairCol, dim);

    // Display GPU run time
    cout << "\ntime GPU: " << t*1000<<" ms" << endl;

    // Free heap memory
    free(cpuToeplitz);
    free(cpuQ);
    free(cpuBlockPairCol);

    return 0;
}


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
#include <vector>
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
    float *cpuBlockPairCol, *cpuQ; 
    float *cpuToeplitz = (float *)malloc(dim*dim*sizeof(*cpuToeplitz));

    // int errorCheck = BuildToeplitz(cpuToeplitz, dim, diagCnt);
    int errorCheck = BuildSymmetricToeplitz(cpuToeplitz, dim, diagCnt);
    if(errorCheck != 0)
    {
        cout << "Issue when executing BuildToeplitz()" << endl;
        return -2;
    }

    cout << endl << "Toeplitz before:" << endl;
    PrintMatrix(cpuToeplitz, dim, dim);

    // Start timer
    Timer timer;
    float t = timer.get();
    timer.start();

    // Compute tridiagonal matrix via GPU
    for(int b = 0; b < (dim - 2); b++)
    {
        cpuBlockPairCol = (float *)malloc((dim - b)*sizeof(*cpuBlockPairCol));
        cpuQ = (float *)malloc((dim - b)*(dim - b)*sizeof(*cpuQ));

        // BlockPairReduction() on specific Toeplitz block pair
        errorCheck = BlockPairReduction(cpuQ, cpuBlockPairCol, cpuToeplitz, dim, b);
        if(errorCheck != 0)
        {
            cout << "Issue when executing BlockPairReduction()" << endl;
            return -3;
        }

        free(cpuQ);
        free(cpuBlockPairCol);
    }

    cout << endl << "Toeplitz after:" << endl;
    PrintMatrix(cpuToeplitz, dim, dim);

    // End timer
    timer.end();  t = timer.get();  // elapsed time in seconds

    // Display GPU run time
    cout << endl << "time GPU: " << t*1000<<" ms" << endl;

    // Free heap memory
    free(cpuToeplitz);

    return 0;
}


// ###
// ###
// ### Symmetric Banded Matrix Reduction to Tridiagonal Form via Householder Transformations
// ### 
// ### 
// ### Approach: Implement Bruno Lang's 2.1 parallel algorithm for tridiagonalization
// ### 
// ###
// ### Grant Bartel, grant.bartel@tum.de
// ### Christoph Riesinger, riesinge@in.tum.de
// ###
// ### 
// ### Technical University of Munich
// ###
// ###

#include "aux.h"
#include "toeplitz.h"
#include "householder.cuh"
#include "MatrixNumerics.h"
#include "Toeplitz.h"
#include "Matrix.h"

#include <iostream>
#include <stdio.h>
#include <vector>

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

    // Confirm validity of parameters
    if(diagCnt >= dim-1)
    {
        cout << "The semi-bandwidth (" << diagCnt << ") must be less than dimension minus one (" << (dim - 1) << ")." << endl;
        return -1;
    }

    Toeplitz<float> *toeplitz = new Toeplitz<float>(dim, diagCnt);
    cout << endl << "Toeplitz before:" << endl;
    toeplitz->Print();

    // Start timer
    Timer timer;
    float t = timer.get();
    timer.start();

    Matrix<float> *triDiagMatrix = MatrixNumerics<float>::LangTridiagonalization21(*toeplitz);

    // End timer
    timer.end();  t = timer.get();  // elapsed time in seconds

    cout << endl << "Toeplitz after:" << endl;
    triDiagMatrix->Print();

    // Display GPU run time
    cout << "time GPU: " << t*1000<<" ms" << endl;

    // Free heap memory
    delete triDiagMatrix;
    delete toeplitz;

    return 0;
}


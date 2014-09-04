// ###
// ###
// ### Symmetric Banded Matrix Reduction to Tridiagonal Form via Householder Transformations
// ### 
// ### 
// ### Approach: Implement Bruno Lang's parallel algorithm for tridiagonalization
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
    /* TEST MATRIX CLASS */
    // Basic instantiation
    // Matrix<float> *testMatrix = new Matrix<float>(10, 10, -1.0);
    // // testMatrix->Print();

    // // Update elements of matrix
    // testMatrix->SetElement(0, 0, 8.0);
    // testMatrix->SetElement(1, 1, 9.0);
    // testMatrix->SetElement(1, 0, 10.0);
    // testMatrix->Print();

    // Matrix<float> *testMatrix1 = new Matrix<float>(10, 1, -1.0);
    // testMatrix1->Print();

    // Matrix<float> *testVector = testMatrix->GetBlock(0, 0, 1, 10);
    // testVector->Print();

    // // Create Toeplitz matrix
    // Toeplitz<float> *testToeplitz = new Toeplitz<float>(7, 2);
    // testToeplitz->Print();

    // // // Retrieve block of matrix
    // Matrix<float> *testBlock = testToeplitz->GetBlock(0, 0, 2, 2);
    // testBlock->Print();

    // // Write to block of Toeplitz
    // testToeplitz->SetBlock(*testBlock, 5, 0);
    // testToeplitz->Print();

    // delete testMatrix;
    // delete testBlock;
    // delete testToeplitz;

    cudaDeviceSynchronize(); CUDA_CHECK;
    
    if(argc != 3)
    {
        cout << "Enter correct amount of arguments {[dim] [diagCnt]}." << endl;
        return -1;
    }

    // Initialize computation parameters
    const int dim = atoi(argv[1]);
    const int diagCnt = atoi(argv[2]);
    // float *cpuToeplitz = (float *)malloc(dim*dim*sizeof(*cpuToeplitz));

    // int errorCheck = BuildSymmetricToeplitz(cpuToeplitz, dim, diagCnt);
    // if(errorCheck != 0)
    // {
    //     cout << "Issue when executing BuildSymmetricToeplitz()" << endl;
    //     return -2;
    // }

    cout << endl << "Toeplitz before:" << endl;
    // PrintMatrix(cpuToeplitz, dim, dim);

    Toeplitz<float> *toeplitz = new Toeplitz<float>(dim, diagCnt);
    toeplitz->Print();

    // Start timer
    Timer timer;
    float t = timer.get();
    timer.start();

    Matrix<float> *triDiagMatrix = MatrixNumerics<float>::LangTridiagonalization21(*toeplitz);

    // errorCheck = BlockPairReduction(cpuToeplitz, dim, diagCnt, 0);
    // if(errorCheck != 0)
    // {
    //     cout << "Issue when executing BlockPairReduction()" << endl;
    //     return -3;
    // }

    // Compute tridiagonal matrix via GPU
    // for(int nu = 0; nu < (dim - 2); nu++)
    // {
    //     errorCheck = BlockPairReduction(cpuToeplitz, dim, diagCnt, nu);
    //     if(errorCheck != 0)
    //     {
    //         cout << "Issue when executing BlockPairReduction()" << endl;
    //         return -3;
    //     }
    // }

    // for(int b = 0; b < (dim - 2); b++)
    // {
    //     cpuBlockPairCol = (float *)malloc((dim - b)*sizeof(*cpuBlockPairCol));
    //     cpuQ = (float *)malloc((dim - b)*(dim - b)*sizeof(*cpuQ));

    //     // BlockPairReduction() on specific Toeplitz block pair
    //     errorCheck = BlockPairReduction(cpuQ, cpuBlockPairCol, cpuToeplitz, dim, b);
    //     if(errorCheck != 0)
    //     {
    //         cout << "Issue when executing BlockPairReduction()" << endl;
    //         return -3;
    //     }

    //     free(cpuQ);
    //     free(cpuBlockPairCol);
    // }

    cout << endl << "Toeplitz after:" << endl;
    // PrintMatrix(cpuToeplitz, dim, dim);
    triDiagMatrix->Print();

    // End timer
    timer.end();  t = timer.get();  // elapsed time in seconds

    // Display GPU run time
    cout << endl << "time GPU: " << t*1000<<" ms" << endl;

    // Free heap memory
    // free(cpuToeplitz);
    delete triDiagMatrix;
    delete toeplitz;

    return 0;
}


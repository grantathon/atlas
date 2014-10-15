// ###
// ###
// ### Symmetric Banded Matrix Reduction to Tridiagonal Form via Householder Transformations
// ### 
// ### 
// ### Approach: Implement Bruno Lang's parallel algorithms for matrix tridiagonalization
// ### 
// ###
// ### Grant Bartel, grant.bartel@tum.de / grant.bartel@gmail.com
// ### Christoph Riesinger, riesinge@in.tum.de
// ###
// ### 
// ### Technical University of Munich
// ### Department of Informatics
// ### Computational Science and Engineering
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
#include <string>

using namespace std;

int main(int argc, char **argv)
{
    // Initialize computation parameters
    int dim = 0;
    int diagCnt = 0;
    Toeplitz<float> *toeplitz;

    // Determine source of matrix data
    switch(argc)
    {
        case 3: // Construct matrix internally with "random" numbers
            dim = atoi(argv[1]);
            diagCnt = atoi(argv[2]);

            // Confirm validity of input parameters
            if(diagCnt >= dim-1)
            {
                cout << "The semi-bandwidth (" << diagCnt << ") must be less than dimension minus one (" << (dim - 1) << ")." << endl;
                return -1;
            }

            toeplitz = new Toeplitz<float>(dim, diagCnt);
            break;
        case 4:  // Construct matrix with input file
            toeplitz = new Toeplitz<float>(argv[3]);

            cout << "GetBandwidth(): " << toeplitz->GetBandwidth() << endl;

            // Confirm validity of input parameter
            if(toeplitz->GetBandwidth() == 0)
            {
                cout << "Issue occured during matrix construction via input file (probably because bad file URI...)" << endl;
                return -1;
            }

            break;
        default:
            cout << "Enter correct amount of arguments {[dim] [diagCnt] [inputFile]} (inputFile is optional)." << endl;
            return -1;
    }
    
    cout << endl << "Matrix before:" << endl;
    toeplitz->Print();

    // Start timer
    Timer timer;
    float t = timer.get();
    timer.start();

    Matrix<float> *triDiagMatrix = MatrixNumerics<float>::LangTridiagonalization21(*toeplitz);

    // End timer
    timer.end();  t = timer.get();  // elapsed time in seconds

    cout << endl << "Matrix after:" << endl;
    triDiagMatrix->Print();

    // Display GPU run time
    cout << "Elapsed time for LangTridiagonalization21: " << t*1000<<" ms" << endl;

    // Free heap memory
    delete triDiagMatrix;
    delete toeplitz;

    return 0;
}


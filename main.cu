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
#include <sstream>
#include <stdlib.h>
#include <stdio.h>

#include "toeplitz.h"

using namespace std;

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        printf("Enter correct amount of arguments (2).\n");
        return -1;
    }

    int dim = atoi(argv[1]);
    int diagCnt = atoi(argv[2]);
    double *toeplitz = (double *)malloc(dim*dim*sizeof(*toeplitz));
    int buildToeplitzError = 0;

    buildToeplitzError = BuildToeplitz(toeplitz, dim, diagCnt);
    if(buildToeplitzError != 0)
    {
        return -2;
    }

    PrintMatrix(toeplitz, dim, dim);

    return 0;
}


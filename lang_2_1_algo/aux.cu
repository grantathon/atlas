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

#include "aux.h"
#include <cstdlib>
#include <iostream>
#include <stdio.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

using std::stringstream;
using std::cerr;
using std::cout;
using std::endl;
using std::string;

// parameter processing: template specialization for T=bool
template<>
bool getParam<bool>(std::string param, bool &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc) || argv[i+1][0]=='-') { var = true; return true; }
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}

// cuda error checking
string prev_file = "";
int prev_line = 0;
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}

void PrintMatrix(float *matrix, int m, int n)
{
    for(unsigned long i = 0; i < (size_t)m; i++)
    {
        printf("|");

        for(unsigned long j = 0; j < (size_t)n; j++)
        {
            const size_t idx = j + m*i;
            printf(" %1.2f", matrix[idx]);
        }

        printf(" |\n");
    }
}

void PrintVector(float *vector, int m)
{
    for(unsigned long i = 0; i < (size_t)m; i++)
    {
        printf("| %1.2f |\n", vector[i]);
    }
}
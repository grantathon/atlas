#ifndef MATRIX_TOEPLITZ_H
#define MATRIX_TOEPLITZ_H

#include "Matrix.h"

#include <iostream>
#include <exception>

using namespace std;

template <class T>
class Toeplitz : Matrix<T>
{
public:
    // Constructors/destructor
    Toeplitz();
    Toeplitz(int m, int n, int bandwidth);
    // Toeplitz(int m, int n, int bandwidth, T* bwElements);
    virtual ~Toeplitz();
};

#endif
#ifndef MATRIX_TOEPLITZ_H
#define MATRIX_TOEPLITZ_H

#include "Matrix.h"

#include <iostream>
#include <exception>
#include <cstdlib>

using namespace std;

template <class T>
class Toeplitz : public Matrix<T>
{
public:
    // Constructors/destructor
    Toeplitz();
    Toeplitz(int dim, int bandwidth);
    virtual ~Toeplitz();
};

#endif
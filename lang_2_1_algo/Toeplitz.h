#ifndef MATRIX_TOEPLITZ_H
#define MATRIX_TOEPLITZ_H

#include "Matrix.h"

#include <iostream>
#include <exception>
#include <cstdlib>
#include <string>

using namespace std;

template <class T>
class Toeplitz : public Matrix<T>
{
private:
    int bandwidth;

public:
    // Constructors/destructor
    Toeplitz();
    Toeplitz(const string& fileURI);
    Toeplitz(int dim, int bandwidth);
    Toeplitz(int dim, int bandwidth, T* data);
    virtual ~Toeplitz();

    // Getters/setters
    int GetBandwidth() const;
    // void SetBandwidth(int bw);
};

#endif
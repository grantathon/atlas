#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <exception>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <string>

#include "boost/foreach.hpp"
#include "boost/tokenizer.hpp"

using namespace std;
using namespace boost;

template <class T>
class Matrix
{
protected:
    int xDim;  // column count
    int yDim;  // row count
    T* matrixData;

public:
    // Constructors/destructor
    Matrix();
    Matrix(const string& fileURI);
    Matrix(int xDim, int yDim);
    Matrix(int xDim, int yDim, T initValue);
    Matrix(int xDim, int yDim, T* data);
    virtual ~Matrix();

    // Getters/setters
    int GetDimX() const;
    int GetDimY() const;
    void ResetDimXY(int xDim, int yDim);

    T* GetMatrixData() const;  // Gets the whole matrix
    // void SetMatrix(const T** data);  // Sets the whole matrix

    Matrix<T>* GetBlock(int x, int y, int xBlockDim, int yBlockDim) const;
    void SetBlock(const Matrix<T>& inputBlock, int xDest, int yDest);

    T* GetBlockData(int x, int y, int xBlockDim, int yBlockDim) const;
    // void SetBlockData(const T* data, int xDest, int yDest);

    T GetElement(int x, int y) const;
    void SetElement(int x, int y, T value);

    // Functions
    void Print() const;
};

#endif
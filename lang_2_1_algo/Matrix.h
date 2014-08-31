#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <exception>

using namespace std;

template <class T>
class Matrix
{
protected:
    int xDim;  // column count
    int yDim;  // row count
    T** matrixData;

public:
    // Constructors/destructor
    Matrix();
    Matrix(int m, int n);
    Matrix(int m, int n, T initValue);
    // Matrix(int m, int n, T** data);
    virtual ~Matrix();

    // Getters/setters
    // T** GetMatrix() const;  // Gets the whole matrix
    // void SetMatrix(const T** data);  // Sets the whole matrix

    Matrix<T>* GetBlock(int x, int y, int xBlockDim, int yBlockDim) const;  // Gets a rectangular/square portion of the matrix from some starting point (x, y)
    void SetBlock(const T** data, int x, int y, int xBlockDim, int yBlockDim);  // Sets the whole matrix

    T GetElement(int x, int y) const;
    void SetElement(int x, int y, T value);

    // Functions
    void Print() const;
};

#endif
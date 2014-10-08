#ifndef MATRIX_NUMERICS_H
#define MATRIX_NUMERICS_H

#include "Matrix.h"
#include "Toeplitz.h"
#include "aux.h"
#include "cublas_v2.h"

#include <iostream>
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))
#define ZERO_TOL 1e-6

template <class T>
class MatrixNumerics
{
private:
    // Constructors/destructor (should never be used)
    MatrixNumerics();
    virtual ~MatrixNumerics();

    static void UpdateColumnRowPair(Matrix<float>& targetMatrix, const Matrix<float>& householderMatrix, const Matrix<float>& vector, int shift);

    static void UpdateMatrixBlock(Matrix<float>& targetMatrix, const Matrix<float>& householderMatrix, int x, int y, int xDim, int yDim);

    static void UpdateMatrixBlockDiagonals(Matrix<float>& targetMatrix, Matrix<float>& householderMatrix, int x, int y, int xDim, int yDim);

public:
    // Functions
    // static Matrix<T>* GetHouseholderMatrix(const Matrix<T>& inputMatrix);

    static Matrix<T>* GetRestrictedHouseholderMatrix(const Matrix<T>& inputMatrix, int householderDim, bool isTriangular);

    static Matrix<T>* LangTridiagonalization21(const Toeplitz<T>& inputMatrix);
};

#endif
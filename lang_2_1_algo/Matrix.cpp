#include "Matrix.h"

template <class T>
Matrix<T>::Matrix()
{
    // PURPOSELY EMPTY
}

template <class T>
Matrix<T>::Matrix(int xDim, int yDim)
{
    this->xDim = xDim;
    this->yDim = yDim;

    // Allocate space for matrix
    matrixData = new T[xDim*yDim];
}

template <class T>
Matrix<T>::Matrix(int xDim, int yDim, T initValue)
{
    this->xDim = xDim;
    this->yDim = yDim;

    // Allocate space for matrix
    matrixData = new T[xDim*yDim];

    // Initialize matrix elements
    for(int y = 0; y < yDim; y++)
    {
        for(int x = 0; x < xDim; x++)
        {
            size_t idx = (size_t)x + (size_t)y*xDim;
            matrixData[idx] = initValue;
        }
    }
}

template <class T>
Matrix<T>::Matrix(int xDim, int yDim, T* data)
{
    this->xDim = xDim;
    this->yDim = yDim;

    // Allocate space for matrix
    matrixData = new T[xDim*yDim];

    // Initialize matrix elements
    for(int y = 0; y < yDim; y++)
    {
        for(int x = 0; x < xDim; x++)
        {
            size_t idx = (size_t)x + (size_t)y*xDim;
            matrixData[idx] = data[idx];
        }
    }
}

template <class T>
Matrix<T>::~Matrix()
{
    delete[] matrixData;
}

template <class T>
int Matrix<T>::GetDimX() const
{
    return xDim;
}

template <class T>
int Matrix<T>::GetDimY() const
{
    return yDim;
}

template <class T>
void Matrix<T>::ResetDimXY(int xDim, int yDim)
{
    this->xDim = xDim;
    this->yDim = yDim;

    // Clear data
    delete[] matrixData;

    // Allocate space for matrix
    matrixData = new T[xDim*yDim];

    // Initialize matrix elements
    for(int y = 0; y < yDim; y++)
    {
        for(int x = 0; x < xDim; x++)
        {
            size_t idx = (size_t)x + (size_t)y*xDim;
            matrixData[idx] = 0.0;
        }
    }
}

template <class T>
T* Matrix<T>::GetMatrixData() const
{
    // return matrixData;

    T *returnMatrixData = new T[xDim*yDim];
    std::memcpy(returnMatrixData, matrixData, (size_t)xDim*yDim*sizeof(T));

    return returnMatrixData;
}

template <class T>
T Matrix<T>::GetElement(int x, int y) const
{
    T element;
    size_t idx = (size_t)x + (size_t)y*xDim;

    try
    {
        element = this->matrixData[idx];
    }
    catch(exception& ex)
    {
        cout << "Exception raised during GetElement(): " << ex.what() << endl;
    }

    return element;
}

template <class T>
void Matrix<T>::SetElement(int x, int y, T value)
{
    size_t idx = (size_t)x + (size_t)y*xDim;

    try
    {
        this->matrixData[idx] = value;
    }
    catch(exception& ex)
    {
        cout << "Exception raised during SetElement(): " << ex.what() << endl;
    }
}

template <class T>
Matrix<T>* Matrix<T>::GetBlock(int x, int y, int xBlockDim, int yBlockDim) const
{
    Matrix<T> *block = new Matrix<T>(xBlockDim, yBlockDim);

    // Check dimension validity
    if((x + xBlockDim) <= this->xDim && (y + yBlockDim) <= this->yDim)
    {
        // Copy values from object matrix to block output
        for(int i = 0; i < yBlockDim; i++)
        {
            for(int j = 0; j < xBlockDim; j++)
            {
                size_t idx = (size_t)(x + j) + (size_t)(y + i)*xDim;
                block->SetElement(j, i, this->matrixData[idx]);
            }  
        }
    }
    else
    {
        cout << "Invalid dimensions passed to GetBlock()" << endl;
    }

    return block;
}

template <class T>
void Matrix<T>::SetBlock(const Matrix<T>& inputBlock, int xDest, int yDest)
{
    // Check dimension validity
    if((xDest + inputBlock.xDim) <= this->xDim && (yDest + inputBlock.yDim) <= this->yDim)
    {
        // Copy values from object matrix to block output
        for(int y = 0; y < inputBlock.yDim; y++)
        {
            for(int x = 0; x < inputBlock.xDim; x++)
            {
                SetElement(x + xDest, y + yDest, inputBlock.GetElement(x, y));
            }  
        }
    }
    else
    {
        cout << "Invalid dimensions passed to GetBlock()" << endl;
    }
}

template <class T>
T* Matrix<T>::GetBlockData(int x, int y, int xBlockDim, int yBlockDim) const
{
    T *blockData = new T[xBlockDim*yBlockDim];

    // Check dimension validity
    if((x + xBlockDim) <= this->xDim && (y + yBlockDim) <= this->yDim)
    {
        // Copy values from object matrix to block output
        for(int i = 0; i < yBlockDim; i++)
        {
            for(int j = 0; j < xBlockDim; j++)
            {
                size_t sinkIdx = (size_t)j + (size_t)i*xBlockDim;
                size_t sourceIdx = (size_t)(x + j) + (size_t)(y + i)*xDim;

                blockData[sinkIdx] = this->matrixData[sourceIdx];
            }  
        }
    }
    else
    {
        cout << "Invalid dimensions passed to GetBlockData()" << endl;
    }

    return blockData;
}

template <class T>
void Matrix<T>::Print() const
{
    for(int y = 0; y < yDim; y++)
    {
        cout << "|";

        for(int x = 0; x < xDim; x++)
        {
            size_t idx = (size_t)x + (size_t)y*xDim;
            cout << " " << setw(6) << setprecision(3) << matrixData[idx];
        }

        cout << " |" << endl;
    }

    cout << endl;
}

// Explicit template instantiations for supported types
template class Matrix<float>;
template class Matrix<double>;
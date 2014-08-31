#include "Matrix.h"

template <class T>
Matrix<T>::Matrix()
{
}

template <class T>
Matrix<T>::Matrix(int m, int n)
{
    yDim = m;
    xDim = n;

    // Allocate space for matrix
    matrixData = new T*[yDim];
    for(int i = 0; i < yDim; i++)
    {
        matrixData[i] = new T[xDim];
    }
}

template <class T>
Matrix<T>::Matrix(int m, int n, T initValue)
{
    yDim = m;
    xDim = n;

    // Allocate space for matrix
    matrixData = new T*[yDim];
    for(int i = 0; i < yDim; i++)
    {
        matrixData[i] = new T[xDim];
    }

    // Initialize matrix elements
    for(int y = 0; y < yDim; y++)
    {
        for(int x = 0; x < xDim; x++)
        {
            matrixData[y][x] = initValue;
        }  
    }
}

// template <class T>
// Matrix<T>::Matrix(int m, int n, T** data)
// {
//     yDim = m;
//     xDim = n;

//     // Allocate space for matrix
//     matrixData = new T*[yDim];
//     for(int i = 0; i < yDim; i++)
//     {
//         matrixData[i] = new T[xDim];
//     }

//     // Set matrix elements
//     for(int y = 0; y < yDim; y++)
//     {
//         for(int x = 0; x < xDim; x++)
//         {
//             matrixData[y][x] = data[y][x];
//         }
//     }
// }

template <class T>
Matrix<T>::~Matrix()
{
    // Deallocate space for matrix
    for(int i = 0; i < yDim; i++)
    {
        delete matrixData[i];
    }

    delete[] matrixData;
}

template <class T>
T Matrix<T>::GetElement(int x, int y) const
{
    T element;

    try
    {
        element = this->matrixData[y][x];
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
    try
    {
        this->matrixData[y][x] = value;
    }
    catch(exception& ex)
    {
        cout << "Exception raised during SetElement(): " << ex.what() << endl;
    }
}

template <class T>
Matrix<T>* Matrix<T>::GetBlock(int x, int y, int xBlockDim, int yBlockDim) const
{
    Matrix<T> *block = new Matrix<T>(yBlockDim, xBlockDim);

    // Check dimension validity
    if((x + xBlockDim) <= this->xDim && (y + yBlockDim) <= this->yDim)
    {
        // Copy values from object matrix to block output
        for(int i = 0; i < yBlockDim; i++)
        {
            for(int j = 0; j < xBlockDim; j++)
            {
                block->SetElement(j, i, this->matrixData[i + y][j + x]);
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
void Matrix<T>::SetBlock(const T** data, int x, int y, int xBlockDim, int yBlockDim)
{
    // Check dimension validity
    if((x + xBlockDim) <= this->xDim && (y + yBlockDim) <= this->yDim)
    {
        // Copy values from object matrix to block output
        for(int i = 0; i < yBlockDim; i++)
        {
            for(int j = 0; j < xBlockDim; j++)
            {
                this->SetElement(j + x, i + y, data[i][j]);
            }  
        }
    }
    else
    {
        cout << "Invalid dimensions passed to GetBlock()" << endl;
    }
}

template <class T>
void Matrix<T>::Print() const
{
    for(int y = 0; y < yDim; y++)
    {
        // printf("|");
        cout << "|";

        for(int x = 0; x < xDim; x++)
        {
            // const size_t idx = j + m*i;
            // printf(" %1.2f", matrix[idx]);
            // cout << " " << setiosflags(ios::fixed) << setprecision(3) << matrix[idx];
            cout << " " << matrixData[y][x];
        }

        // printf(" |\n");
        cout << " |" << endl;
    }

    cout << endl;
}

// Explicit template instantiations for supported types
template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
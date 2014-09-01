#include "Toeplitz.h"

template <class T>
Toeplitz<T>::Toeplitz()
{
    // PURPOSELY EMPTY
}

template <class T>
Toeplitz<T>::Toeplitz(int dim, int bandwidth)
    : Matrix<T>::Matrix(dim, dim)
{
    size_t idx = 0;
    int absXY = 0;

    // Determine parameter validity
    if(bandwidth >= dim)
    {
        cout << "The bandwidth is larger than the m and/or n" << endl;
    }
    
    float *diagConsts = new float[bandwidth];

    // Assigns "random" numbers to the diagonal constants
    for(size_t i = 0; i < (size_t)bandwidth + 1; i++)
    {
        diagConsts[i] = ((float)rand() / (float)RAND_MAX);
    }

    // Construct the Toeplitz matrix
    for(size_t y = 0; y < (size_t)dim; y++)
    {
        for(size_t x = 0; x < (size_t)dim; x++)
        {
            absXY = abs((int)x-(int)y);
            idx = (size_t)x + (size_t)y*this->xDim;

            if(absXY > bandwidth)  // Bound the bandwidth with zeros
            {
                this->matrixData[idx] = 0.0;
            }
            else
            {
                this->matrixData[idx] = diagConsts[absXY];
            }
        }
    }
    
    delete diagConsts;
}

template <class T>
Toeplitz<T>::~Toeplitz()
{

}

// Explicit template instantiations for supported types
template class Toeplitz<int>;
template class Toeplitz<float>;
template class Toeplitz<double>;
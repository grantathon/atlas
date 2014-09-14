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
    this->bandwidth = bandwidth;

    size_t idx = 0;
    int absXY = 0;

    // Determine parameter validity
    if(bandwidth >= dim - 1)
    {
        cout << "The bandwidth is larger than the matrix dimension" << endl;
    }
    
    float *diagConsts = new float[bandwidth + 1];

    // Assigns "random" numbers to the diagonal constants
    // for(size_t i = 0; i < (size_t)bandwidth; i++)
    for(size_t i = 0; i < (size_t)bandwidth + 1; i++)
    {
        diagConsts[i] = ((float)rand() / (float)RAND_MAX);
        cout << diagConsts[i] << " ";
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
    
    delete[] diagConsts;
}

template <class T>
Toeplitz<T>::Toeplitz(int dim, int bandwidth, T* data)
    : Matrix<T>::Matrix(dim, dim, data)
{
    this->bandwidth = bandwidth;

    // Determine parameter validity
    if(bandwidth >= dim - 1)
    {
        cout << "The bandwidth is larger than the matrix dimension" << endl;
    }
}

template <class T>
Toeplitz<T>::~Toeplitz()
{

}

template <class T>
int Toeplitz<T>::GetBandwidth() const
{
    return bandwidth;
}

// template <class T>
// void Toeplitz<T>::SetBandwidth(int bw)
// {
//     bandwidth = bw;
// }

// Explicit template instantiations for supported types
// template class Toeplitz<int>;
template class Toeplitz<float>;
template class Toeplitz<double>;
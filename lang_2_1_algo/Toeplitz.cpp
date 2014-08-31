#include "Toeplitz.h"

template <class T>
Toeplitz<T>::Toeplitz()
{
}

template <class T>
Toeplitz<T>::Toeplitz(int m, int n, int bandwidth)
    : Matrix<T>::Matrix(m, n)
{
    size_t idx = 0;
    int absXY = 0;

    // Determine parameter validity
    if(bandwidth >= m || bandwidth >= n)
    {
        cout << "The bandwidth is larger than the m and/or n" << endl;
    }
    
    float *diagConsts = new float[bandwidth];
    // float *diagConsts = (float *)malloc(r*sizeof(*diagConsts));

    // Assigns "random" numbers to the diagonal constants
    for(int i = 0; i < bandwidth + 1; i++)
    {
        diagConsts[i] = ((float)rand() / (float)RAND_MAX);
    }

    for(size_t y = 0; y < (size_t)dim; y++)
    {
        for(size_t x = 0; x < (size_t)dim; x++)
        {
            idx = x + (size_t)dim*y;
            absXY = abs((int)x-(int)y);

            if(absXY > r)
            {
                t_matrix[idx] = 0.0;
            }
            else
            {
                t_matrix[idx] = diagConsts[absXY];
            }
        }
    }
    
    delete diagConsts;
}

// template <class T>
// Toeplitz<T>::Toeplitz(int m, int n, int bandwidth, T* bwElements)
//     : Matrix<T>::Matrix(m, n)
// {

// }

template <class T>
Toeplitz<T>::~Toeplitz()
{

}

// Explicit template instantiations for supported types
template class Toeplitz<int>;
template class Toeplitz<float>;
template class Toeplitz<double>;
#include "MatrixNumerics.h"

template <class T>
MatrixNumerics<T>::MatrixNumerics()
{
    // PURPOSELY EMPTY
}

template <class T>
MatrixNumerics<T>::~MatrixNumerics()
{
    // PURPOSELY EMPTY
}

/* Prepare Q matrix for Householder transformation */
template <class T>
__global__ void ComputeRestrictedHouseholderMatrix(T *householderMatrix, T *householderVector, int dim)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(x < dim && y < dim)
    {
        T x_val = householderVector[x];
        T y_val = householderVector[y];

        if(x == y)  // Compute diagonal value
        {
            householderMatrix[IDX2C(y, x, dim)] = 1.0 - 2.0*powf(fabs(x_val), 2.0);
        }
        else  // Compute symmetric values
        {
            householderMatrix[IDX2C(y, x, dim)] = -2.0*x_val*y_val;
        }
    }
}

template <class T>
__global__ void ComputeRestrictedHouseholderMatrix(T *inputMatrix, T *householderMatrix, int dim, int dummy)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    
    if(x < dim && y < dim)
    {
        size_t idx = (size_t)x + (size_t)dim*y;

        // Compute local alpha element for the householder vector
        T *alpha;
        *alpha = 0.0;
        T localAlpha = 0.0;
        if(x == 0 && y > 0)
        {
            localAlpha = powf(fabs(gInputMatrix[idx]), 2);  
        }

        atomicAdd(alpha, localAlpha);
        __syncthreads();

        // extern __shared__ T sAlphas[];
        // size_t tid = threadIdx.x;
        // if(x == 0 && y > 0)
        // {
        //     sAlphas[tid] = powf(fabs(gInputMatrix[idx]), 2);  
        // }
        // else
        // {
        //     sAlphas[tid]  = 0.0;
        // }

        // // Sum all alphas across all threads
        // __syncthreads();
        // for(int i = 0; i < dim*dim; i++)
        // {

        // }

        // Find rho for Householder vector computation


        // T x_val = householderVector[x];
        // T y_val = householderVector[y];

        if(x == y)  // Compute diagonal value
        {
            householderMatrix[IDX2C(y, x, dim)] = 1.0 - 2.0*powf(fabs(x_val), 2.0);
        }
        else  // Compute symmetric values
        {
            householderMatrix[IDX2C(y, x, dim)] = -2.0*x_val*y_val;
        }
    }
}

template <class T>
Matrix<T>* MatrixNumerics<T>::GetRestrictedHouseholderMatrix(const Matrix<T>& columnVector, int householderDim, bool isTriangular)
{
    float alpha = 0.0;
    float rho = 0.0;
    int dim = householderDim;
    Matrix<T> *householderVector = new Matrix<T>(1, dim, 0.0);
    Matrix<T> *householderMatrix = new Matrix<T>(dim, dim, 0.0);

    // Compute alpha parameter for the householder vector
    for(int i = 0; i < dim; i++)
    {
        alpha += powf(fabs(columnVector.GetElement(0, i)), 2);
    }

    // Determine the type of householder vector to construct
    if(isTriangular)
    {
        // Determine sign of alpha
        if(columnVector.GetElement(0, 0) > 0)
        {
            alpha = sqrtf(alpha);
        }
        else
        {
            alpha = -sqrtf(alpha); 
        }

        // Initialize contents of householder vector
        householderVector->SetElement(0, 0, columnVector.GetElement(0,0) + alpha);
        for(int i = 1; i < dim; i++)
        {
            householderVector->SetElement(0, i, columnVector.GetElement(0, i));
        }

        // Compute rho parameter for the householder vector (i.e., 2-norm)
        for(int i = 0; i < dim; i++)
        {
            rho += powf(fabs(householderVector->GetElement(0, i)), 2);
        }
        rho = sqrtf(rho);

        // Finish constructing the householder vector
        for(int i = 0; i < dim; i++)
        {
            householderVector->SetElement(0, i, householderVector->GetElement(0, i) / rho);
        }
    }
    else
    {
        // Determine sign of alpha
        if(columnVector.GetElement(0, 0) > 0)
        {
            alpha = -sqrtf(alpha);
        }
        else
        {
            alpha = sqrtf(alpha); 
        }

        // Compute rho parameter for the householder vector
        rho = sqrtf((powf(alpha, 2) - columnVector.GetElement(0, 0)*alpha) / 2);

        // Construct the householder vector
        householderVector->SetElement(0, 0, (columnVector.GetElement(0, 0) - alpha) / (2*rho));
        for(int i = 1; i < dim; i++)
        {
            householderVector->SetElement(0, i, columnVector.GetElement(0, i) / (2*rho));
        }
    }

    // Setup GPU and compute the householder matrix
    T *gHouseholderMatrix, *gHouseholderVector;
    T *householderMatrixData = new T[dim*dim];
    cudaMalloc(&gHouseholderMatrix, (size_t)dim*dim*sizeof(*gHouseholderMatrix)); CUDA_CHECK;
    cudaMalloc(&gHouseholderVector, (size_t)dim*sizeof(*gHouseholderVector)); CUDA_CHECK;
    cudaMemset(gHouseholderMatrix, 0, (size_t)dim*dim*sizeof(*gHouseholderMatrix)); CUDA_CHECK;
    cudaMemcpy(gHouseholderVector, householderVector->GetMatrixData(), (size_t)dim*sizeof(*gHouseholderVector), cudaMemcpyHostToDevice); CUDA_CHECK;
    delete householderVector;

    dim3 block = dim3(32, 32, 1);
    dim3 grid = dim3(((dim + 1) + block.x-1)/block.x, ((dim + 1) + block.y-1)/block.y, 1);

    ComputeRestrictedHouseholderMatrix<T><<<grid, block>>>(gHouseholderMatrix, gHouseholderVector, dim);
    cudaMemcpy(householderMatrixData, gHouseholderMatrix, (size_t)dim*dim*sizeof(*gHouseholderMatrix), cudaMemcpyDeviceToHost); CUDA_CHECK;

    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            float element = householderMatrixData[IDX2C(j, i, dim)];
         
            // Assign the new matrix element if greater than some zero tolerance
            if(fabs(element) > ZERO_TOL)
            {
                householderMatrix->SetElement(j, i, element);
            }
            else
            {
                householderMatrix->SetElement(j, i, 0.0);
            }
        }  
    }

    // Free memory
    cudaFree(gHouseholderMatrix);
    cudaFree(gHouseholderVector);
    delete[] householderMatrixData;

    return householderMatrix;
}

template <class T>
Matrix<T>* MatrixNumerics<T>::GetRestrictedHouseholderMatrix(T *gInputMatrix, int shift, int householderDim, bool isTriangular)
{
    float alpha = 0.0;
    float rho = 0.0;
    int dim = householderDim;
    // Matrix<T> *householderVector = new Matrix<T>(1, dim, 0.0);
    Matrix<T> *householderMatrix = new Matrix<T>(dim, dim, 0.0);

    // Compute alpha parameter for the householder vector
    // for(int i = 0; i < dim; i++)
    // {
    //     alpha += powf(fabs(columnVector.GetElement(0, i)), 2);
    // }

    // // Determine the type of householder vector to construct
    // if(isTriangular)
    // {
    //     // Determine sign of alpha
    //     if(columnVector.GetElement(0, 0) > 0)
    //     {
    //         alpha = sqrtf(alpha);
    //     }
    //     else
    //     {
    //         alpha = -sqrtf(alpha); 
    //     }

    //     // Initialize contents of householder vector
    //     householderVector->SetElement(0, 0, columnVector.GetElement(0,0) + alpha);
    //     for(int i = 1; i < dim; i++)
    //     {
    //         householderVector->SetElement(0, i, columnVector.GetElement(0, i));
    //     }

    //     // Compute rho parameter for the householder vector (i.e., 2-norm)
    //     for(int i = 0; i < dim; i++)
    //     {
    //         rho += powf(fabs(householderVector->GetElement(0, i)), 2);
    //     }
    //     rho = sqrtf(rho);

    //     // Finish constructing the householder vector
    //     for(int i = 0; i < dim; i++)
    //     {
    //         householderVector->SetElement(0, i, householderVector->GetElement(0, i) / rho);
    //     }
    // }
    // else
    // {
    //     // Determine sign of alpha
    //     if(columnVector.GetElement(0, 0) > 0)
    //     {
    //         alpha = -sqrtf(alpha);
    //     }
    //     else
    //     {
    //         alpha = sqrtf(alpha); 
    //     }

    //     // Compute rho parameter for the householder vector
    //     rho = sqrtf((powf(alpha, 2) - columnVector.GetElement(0, 0)*alpha) / 2);

    //     // Construct the householder vector
    //     householderVector->SetElement(0, 0, (columnVector.GetElement(0, 0) - alpha) / (2*rho));
    //     for(int i = 1; i < dim; i++)
    //     {
    //         householderVector->SetElement(0, i, columnVector.GetElement(0, i) / (2*rho));
    //     }
    // }

    // Setup GPU and compute the householder matrix
    T *gHouseholderMatrix; //, *gHouseholderVector;
    T *householderMatrixData = new T[dim*dim];
    cudaMalloc(&gHouseholderMatrix, (size_t)dim*dim*sizeof(*gHouseholderMatrix)); CUDA_CHECK;
    // cudaMalloc(&gHouseholderVector, (size_t)dim*sizeof(*gHouseholderVector)); CUDA_CHECK;
    cudaMemset(gHouseholderMatrix, 0, (size_t)dim*dim*sizeof(*gHouseholderMatrix)); CUDA_CHECK;
    // cudaMemcpy(gHouseholderVector, householderVector->GetMatrixData(), (size_t)dim*sizeof(*gHouseholderVector), cudaMemcpyHostToDevice); CUDA_CHECK;
    // delete householderVector;

    dim3 block = dim3(32, 32, 1);
    dim3 grid = dim3(((dim + 1) + block.x-1)/block.x, ((dim + 1) + block.y-1)/block.y, 1);

    ComputeRestrictedHouseholderMatrix<T><<<grid, block>>>(gInputMatrix, gHouseholderMatrix, dim, 1);
    cudaMemcpy(householderMatrixData, gHouseholderMatrix, (size_t)dim*dim*sizeof(*gHouseholderMatrix), cudaMemcpyDeviceToHost); CUDA_CHECK;

    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            float element = householderMatrixData[IDX2C(j, i, dim)];
         
            // Assign the new matrix element if greater than some zero tolerance
            if(fabs(element) > ZERO_TOL)
            {
                householderMatrix->SetElement(j, i, element);
            }
            else
            {
                householderMatrix->SetElement(j, i, 0.0);
            }
        }  
    }

    // Free memory
    cudaFree(gHouseholderMatrix);
    // cudaFree(gHouseholderVector);
    delete[] householderMatrixData;

    return householderMatrix;
}

template <>
void MatrixNumerics<float>::UpdateColumnRowPair(Matrix<float>& targetMatrix, const Matrix<float>& householderMatrix,
    const Matrix<float>& vector, int shift)
{
    // Setup GPU for cuBLAS and compute the new colum/row pair
    int dim = vector.GetDimY();
    float alpha = 1.0f;
    float beta = 0.0f;
    float *cuHouseholderMatrix, *cuInputVector, *cuOutputVector, *outputVector ;
    cublasHandle_t handle;
    
    // Allocate memory
    outputVector = new float[dim];
    cudaMalloc(&cuHouseholderMatrix, (size_t)dim*dim*sizeof(*cuHouseholderMatrix)); CUDA_CHECK;
    cudaMalloc(&cuInputVector, (size_t)dim*sizeof(*cuInputVector)); CUDA_CHECK;
    cudaMalloc(&cuOutputVector, (size_t)dim*sizeof(*cuOutputVector)); CUDA_CHECK;
   
    // Create cuBLAS handle
    cublasCreate(&handle);
    
    // Initialize matrix and vector
    cublasSetMatrix(dim, dim, sizeof(*householderMatrix.GetMatrixData()), householderMatrix.GetMatrixData(), dim, cuHouseholderMatrix, dim);
    cublasSetVector(dim, sizeof(*vector.GetMatrixData()), vector.GetMatrixData(), 1, cuInputVector, 1);

    // Perform block pair computations
    cublasSgemv(handle, CUBLAS_OP_N, dim, dim, &alpha, cuHouseholderMatrix, dim, cuInputVector, 1, &beta, cuOutputVector, 1);
    
    // Retreive block pair matrix
    cublasGetVector(dim, sizeof(*outputVector), cuOutputVector, 1, outputVector, 1);

    // Copy over new row/column pair elements
    for(int i = 0; i < dim; i++)
    {
        float element = outputVector[IDX2C(0, i, 1)];
     
        // Assign the new matrix element if greater than some zero tolerance
        if(fabs(element) > ZERO_TOL)
        {
            targetMatrix.SetElement(shift, i + shift + 1, element);
            targetMatrix.SetElement(i + shift + 1, shift, element);
        }
        else
        {
            targetMatrix.SetElement(shift, i + shift + 1, 0.0);
            targetMatrix.SetElement(i + shift + 1, shift, 0.0);
        }
    }

    // Free memory
    cudaFree(cuHouseholderMatrix);
    cudaFree(cuInputVector);
    cudaFree(cuOutputVector);
    delete[] outputVector;
    cublasDestroy(handle);
}

template <>
void MatrixNumerics<float>::UpdateMatrixBlock(Matrix<float>& targetMatrix, const Matrix<float>& householderMatrix, int x, int y, int xDim, int yDim)
{
    // Setup GPU for cuBLAS and compute the new colum/row pair
    int dim = householderMatrix.GetDimY();
    float alpha = 1.0f;
    float beta = 0.0f;
    float *cuHouseholderMatrix, *cuAMatrix, *cuTempMatrix, *aMatrixData;
    cublasHandle_t handle;
    
    // Allocate memory
    aMatrixData = targetMatrix.GetBlockData(x, y, xDim, yDim);
    cudaMalloc(&cuHouseholderMatrix, (size_t)dim*dim*sizeof(*cuHouseholderMatrix)); CUDA_CHECK;
    cudaMalloc(&cuAMatrix, (size_t)dim*dim*sizeof(*cuAMatrix)); CUDA_CHECK;
    cudaMalloc(&cuTempMatrix, (size_t)dim*dim*sizeof(*cuTempMatrix)); CUDA_CHECK;

    // TEST
    // Matrix<float> *test = new Matrix<float>(xDim, yDim, aMatrixData);
    // test->Print();
    // delete test;
   
    // Create cuBLAS handle
    cublasCreate(&handle);
    
    // Initialize matrices
    cublasSetMatrix(dim, dim, sizeof(*householderMatrix.GetMatrixData()), householderMatrix.GetMatrixData(), dim, cuHouseholderMatrix, dim);
    cublasSetMatrix(dim, dim, sizeof(*aMatrixData), aMatrixData, dim, cuAMatrix, dim);

    // Perform matrix block computations
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim, &alpha, cuHouseholderMatrix, dim, cuAMatrix, dim, &beta, cuTempMatrix, dim);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, cuTempMatrix, dim, cuHouseholderMatrix, dim, &beta, cuAMatrix, dim);

    // Retreive computed block matrix
    cublasGetMatrix(dim, dim, sizeof(*aMatrixData), cuAMatrix, dim, aMatrixData, dim);

    // Copy new computed elements to target matrix
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            float element = aMatrixData[IDX2C(i, j, dim)];

            if(fabs(element) > ZERO_TOL)
            {
                targetMatrix.SetElement(x + j, y + i, element);
            }
            else
            {
                targetMatrix.SetElement(x + j, y + i, 0.0);
            }
        }  
    }

    // Free memory
    cudaFree(cuHouseholderMatrix);
    cudaFree(cuAMatrix);
    cudaFree(cuTempMatrix);
    delete[] aMatrixData;
    cublasDestroy(handle);
}

template <>
void MatrixNumerics<float>::UpdateMatrixBlockDiagonals(Matrix<float>& targetMatrix, Matrix<float>& householderMatrix,
    int x, int y, int xDim, int yDim)
{
    // Setup GPU for cuBLAS and compute the new colum/row pair
    int dim = householderMatrix.GetDimY();
    float alpha = 1.0f;
    float beta = 0.0f;
    float *cuHouseholderMatrix, *cuOldAMatrix, *cuNewAMatrix, *aMatrixData, *cuNewHouseholderMatrix,
        *cuOutputMatrix, *outputHouseholderMatrixData, *newAMatrix;
    Matrix<float> *aMatrixColumn, *newHouseholderMatrix;
    cublasHandle_t handle;

    aMatrixData = new float[xDim*yDim];
    for(int i = 0; i < yDim; i++)
    {
        for(int j = 0; j < xDim; j++)
        {
            aMatrixData[IDX2C(i, j, yDim)] = targetMatrix.GetElement(x + j, y + i);
        }
    }
    // std::cout << "0.5" << std::endl;

    // Allocate memory
    cudaMalloc(&cuHouseholderMatrix, (size_t)dim*dim*sizeof(*cuHouseholderMatrix)); CUDA_CHECK;
    cudaMalloc(&cuOldAMatrix, (size_t)xDim*yDim*sizeof(*cuOldAMatrix)); CUDA_CHECK;
    cudaMalloc(&cuNewAMatrix, (size_t)xDim*yDim*sizeof(*cuNewAMatrix)); CUDA_CHECK;
   
    // Create cuBLAS handle
    cublasCreate(&handle);
    
    // Initialize matrices
    cublasSetMatrix(dim, dim, sizeof(*householderMatrix.GetMatrixData()), householderMatrix.GetMatrixData(), dim, cuHouseholderMatrix, dim);
    cublasSetMatrix(yDim, xDim, sizeof(*aMatrixData), aMatrixData, yDim, cuOldAMatrix, yDim);

    // std::cout << "1" << std::endl;

    // Compute new matrix that needs first column minimized by a new householder matrix
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, yDim, xDim, xDim, &alpha, cuOldAMatrix, yDim, cuHouseholderMatrix, xDim, &beta, cuNewAMatrix, yDim);

    // std::cout << "2" << std::endl;

    // Retreive new block diagonal matrix
    cublasGetMatrix(yDim, xDim, sizeof(*aMatrixData), cuNewAMatrix, yDim, aMatrixData, yDim);

    // std::cout << "3" << std::endl;

    // Retrieve first column of new block diagonal
    aMatrixColumn = new Matrix<float>(1, yDim, 0.0);
    for(int i = 0; i < yDim; i++)
    {
        aMatrixColumn->SetElement(0, i, aMatrixData[IDX2C(i, 0, xDim)]);
    }
    delete[] aMatrixData;

    // std::cout << "3.5" << std::endl;

    // Adjust dimensions of householder matrix and following computations
    householderMatrix.ResetDimXY(yDim, yDim);
    outputHouseholderMatrixData = new float[yDim*yDim];

    // std::cout << "3.75" << std::endl;

    // Compute the new householder matrix
    newHouseholderMatrix = MatrixNumerics<float>::GetRestrictedHouseholderMatrix(*aMatrixColumn, yDim, true);
    delete aMatrixColumn;
    // newHouseholderMatrix->Print();

    // Set GPU householder matrix to updated version
    cudaMalloc(&cuOutputMatrix, (size_t)yDim*xDim*sizeof(*cuOutputMatrix)); CUDA_CHECK;
    cudaMalloc(&cuNewHouseholderMatrix, (size_t)yDim*yDim*sizeof(*cuNewHouseholderMatrix)); CUDA_CHECK;
    cublasSetMatrix(yDim, yDim, sizeof(*newHouseholderMatrix->GetMatrixData()), newHouseholderMatrix->GetMatrixData(), yDim, cuNewHouseholderMatrix, yDim);
    delete newHouseholderMatrix;

    // Update block diagonal elements in target matrix
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, yDim, xDim, yDim, &alpha, cuNewHouseholderMatrix, yDim, cuNewAMatrix, yDim, &beta, cuOutputMatrix, yDim);

    // std::cout << "4" << std::endl;

    // Retrieve new block diagonal elements and householder matrix
    newAMatrix = new float[yDim*xDim];
    cublasGetMatrix(yDim, xDim, sizeof(*newAMatrix), cuOutputMatrix, yDim, newAMatrix, yDim);
    cublasGetMatrix(yDim, yDim, sizeof(*outputHouseholderMatrixData), cuNewHouseholderMatrix, yDim, outputHouseholderMatrixData, yDim);

    // std::cout << "5" << std::endl;

    // Copy new computed elements to target matrix (required due to matrix indexing differences)
    // as well as set "zero" elements where necessary
    for(int i = 0; i < yDim; i++)
    {
        for(int j = 0; j < xDim; j++)
        {
            float element = newAMatrix[IDX2C(i, j, yDim)];

            // std::cout << element << std::endl;
         
            // Assign the new matrix element if greater than some zero tolerance
            // for the target matrix
            if(fabs(element) > ZERO_TOL)
            {
                targetMatrix.SetElement(x + j, y + i, element);
                targetMatrix.SetElement(y + i, x + j, element);
            }
            else
            {
                targetMatrix.SetElement(x + j, y + i, 0.0);
                targetMatrix.SetElement(y + i, x + j, 0.0);
            }
        }  
    }
    delete[] newAMatrix;

    // std::cout << "6" << std::endl;

    for(int i = 0; i < yDim; i++)
    {
        for(int j = 0; j < yDim; j++)
        {
            float element = outputHouseholderMatrixData[IDX2C(i, j, yDim)];

            // std::cout << element << std::endl;

            // Assign the new matrix element if greater than some zero tolerance
            // for the householder matrix
            if(fabs(element) > ZERO_TOL)
            {
                householderMatrix.SetElement(j, i, element);
            }
            else
            {
                householderMatrix.SetElement(j, i, 0.0);
            }
        }  
    }

    // std::cout << "7" << std::endl;

    cudaDeviceSynchronize();

    // Free memory
    cudaFree(cuHouseholderMatrix);
    cudaFree(cuNewHouseholderMatrix);
    cudaFree(cuOldAMatrix);
    cudaFree(cuNewAMatrix);
    cudaFree(cuOutputMatrix);
    delete[] outputHouseholderMatrixData;
    cublasDestroy(handle);

    // std::cout << "8" << std::endl;
}

template <class T>
Matrix<T>* MatrixNumerics<T>::LangTridiagonalization21(const Toeplitz<T>& matrix)
{
    Toeplitz<T> *outputMatrix;

    // Only run against square matrices
    if(matrix.GetDimX() == matrix.GetDimY())
    {
        int baseBlocks, remBlocks;
        int dim = matrix.GetDimX();  // Dimension of matrix
        int bandwidth = matrix.GetBandwidth();  // Bandwidth of square, banded matrix
        outputMatrix = new Toeplitz<T>(dim, bandwidth, matrix.GetMatrixData());

        // Allocate device space for input matrix
        T *gInputMatrix;
        cudaMalloc(&gInputMatrix, (size_t)matrix.GetDimX()*matrix.GetDimY()*sizeof(*gInputMatrix)); CUDA_CHECK;
        cudaMemcpy(gInputMatrix, matrix.GetMatrixData(), (size_t)matrix.GetDimX()*matrix.GetDimY()*sizeof(*gInputMatrix), cudaMemcpyHostToDevice); CUDA_CHECK;

        // std::cout << "dimension = " << dim << std::endl;
        // std::cout << "bandwidth = " << bandwidth << std::endl;

        // Main loop for transforming matrix into tridiagonal form
        for(int nu = 0; nu < dim - 2; nu++)
        {
            // Determine the b & r iteration parameters
            baseBlocks = (dim - nu - 1) / bandwidth;
            remBlocks = dim - nu - bandwidth*(baseBlocks - 1) - 1;
            // std::cout << "baseBlocks = " << baseBlocks << std::endl;
            // std::cout << "remBlocks = " << remBlocks << std::endl;

            if(remBlocks > bandwidth)
            {
                baseBlocks += (remBlocks / bandwidth);
                remBlocks -= bandwidth*(remBlocks / bandwidth);
                // std::cout << "baseBlocks = " << baseBlocks << std::endl;
                // std::cout << "remBlocks = " << remBlocks << std::endl;
            }

            // Set block dimension manually on single matrix computation cycles (i.e., b == 1)
            if(baseBlocks == 1)
            {
                bandwidth = remBlocks;
            }
            else
            {
                bandwidth = matrix.GetBandwidth();
            }

            // Compute the Householder matrix
            Matrix<T> *column = outputMatrix->GetBlock(nu, nu + 1, 1, bandwidth);
            // column->Print();
            // outputMatrix->Print();
            // Matrix<T> *householderMatrix = GetRestrictedHouseholderMatrix(*column, bandwidth, false);
            Matrix<T> *householderMatrix = GetRestrictedHouseholderMatrix(gInputMatrix, nu, bandwidth, false);

            // Update the colum/row pair of the main block
            MatrixNumerics<T>::UpdateColumnRowPair(*outputMatrix, *householderMatrix, *column, nu);
            // outputMatrix->Print();
            // householderMatrix->Print();

            // Update the remaining matrix blocks
            int betaX, betaY, betaDimX, betaDimY;
            for(int beta = 0; beta < baseBlocks; beta++)
            {
                if(beta != baseBlocks - 1)
                {
                    betaX = 2*(beta + 1) + nu - 1;
                    betaY = betaX;
                    betaDimX = bandwidth;
                    betaDimY = betaDimX;

                    // std::cout << "3betaX = " << betaX << std::endl;
                    // std::cout << "betaY = " << betaY << std::endl;
                    // std::cout << "betaDimX = " << betaDimX << std::endl;
                    // std::cout << "betaDimY = " << betaDimY << std::endl;

                    MatrixNumerics<T>::UpdateMatrixBlock(*outputMatrix, *householderMatrix, betaX, betaY, betaDimX, betaDimY);
                    // outputMatrix->Print();
                    // householderMatrix->Print();
                }
                else
                {
                    if(baseBlocks == 1)
                    {
                        betaX = dim - bandwidth;
                        betaY = betaX;
                        betaDimX = remBlocks;
                        betaDimY = betaDimX;
                        // betaX = nu + 1;
                        // betaY = nu + 1;
                        // betaDimX = remBlocks;
                        // betaDimY = betaDimX;
                    }
                    else
                    {
                        betaX += bandwidth;

                        betaDimX = betaDimY;
                        betaDimY = betaDimX;
                    }

                    // std::cout << "2betaX = " << betaX << std::endl;
                    // std::cout << "betaY = " << betaY << std::endl;
                    // std::cout << "betaDimX = " << betaDimX << std::endl;
                    // std::cout << "betaDimY = " << betaDimY << std::endl;

                    MatrixNumerics<T>::UpdateMatrixBlock(*outputMatrix, *householderMatrix, betaX, betaY, betaDimX, betaDimY);
                    // outputMatrix->Print();
                    // householderMatrix->Print();
                }

                if(beta < baseBlocks - 2)
                {
                    betaY += bandwidth;

                    // std::cout << "4betaX = " << betaX << std::endl;
                    // std::cout << "betaY = " << betaY << std::endl;
                    // std::cout << "betaDimX = " << betaDimX << std::endl;
                    // std::cout << "betaDimY = " << betaDimY << std::endl;

                    MatrixNumerics<T>::UpdateMatrixBlockDiagonals(*outputMatrix, *householderMatrix, betaX, betaY, betaDimX, betaDimY);
                    // outputMatrix->Print();
                    // householderMatrix->Print();
                }
                else if(beta == baseBlocks - 2)
                {
                    betaY += bandwidth;
                    betaDimY = remBlocks;

                    // std::cout << "1betaX = " << betaX << std::endl;
                    // std::cout << "betaY = " << betaY << std::endl;
                    // std::cout << "betaDimX = " << betaDimX << std::endl;
                    // std::cout << "betaDimY = " << betaDimY << std::endl;

                    MatrixNumerics<T>::UpdateMatrixBlockDiagonals(*outputMatrix, *householderMatrix, betaX, betaY, betaDimX, betaDimY);
                    // outputMatrix->Print();
                    // householderMatrix->Print();

                    betaDimX = remBlocks;
                }
            }

            delete column;
            delete householderMatrix;
        }
    }
    else
    {
        // outputMatrix = new Matrix<T>(matrix.GetDimX(), matrix.GetDimY(), -1.0);
    }

    return outputMatrix;
}

// Explicit template instantiations for supported types
template class MatrixNumerics<float>;
// template class MatrixNumerics<double>;
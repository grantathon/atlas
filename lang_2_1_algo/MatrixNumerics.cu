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
Matrix<T>* MatrixNumerics<T>::GetRestrictedHouseholderMatrix(const Matrix<T>& columnVector, bool isTriangular)
{
    float alpha = 0.0;
    float rho = 0.0;
    int dim = columnVector.GetDimY();
    Matrix<T> *householderVector = new Matrix<T>(1, dim, 0.0);
    Matrix<T> *householderMatrix;
    // columnVector.Print();

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
    // householderVector->Print();

    // Setup GPU and compute the householder matrix
    T *gHouseholderMatrix, *gHouseholderVector;
    T *householderMatrixData = new T[dim*dim];
    cudaMalloc(&gHouseholderMatrix, (size_t)dim*dim*sizeof(*gHouseholderMatrix)); CUDA_CHECK;
    cudaMalloc(&gHouseholderVector, (size_t)dim*sizeof(*gHouseholderVector)); CUDA_CHECK;
    cudaMemset(gHouseholderMatrix, 0, (size_t)dim*dim*sizeof(*gHouseholderMatrix)); CUDA_CHECK;
    cudaMemcpy(gHouseholderVector, householderVector->GetMatrixData(), (size_t)dim*sizeof(*gHouseholderVector), cudaMemcpyHostToDevice); CUDA_CHECK;

    dim3 block = dim3(32, 32, 1);
    dim3 grid = dim3(((dim + 1) + block.x-1)/block.x, ((dim + 1) + block.y-1)/block.y, 1);

    ComputeRestrictedHouseholderMatrix<T><<<grid, block>>>(gHouseholderMatrix, gHouseholderVector, dim);
    cudaMemcpy(householderMatrixData, gHouseholderMatrix, (size_t)dim*dim*sizeof(*gHouseholderMatrix), cudaMemcpyDeviceToHost); CUDA_CHECK;

    householderMatrix = new Matrix<T>(dim, dim, householderMatrixData);
    // householderMatrix->Print();

    // Free memory
    cudaFree(gHouseholderMatrix);
    cudaFree(gHouseholderVector);
    delete householderMatrixData, householderVector;

    return householderMatrix;
}

template <>
void MatrixNumerics<float>::UpdateColumnRowPair(Matrix<float>& targetMatrix, const Matrix<float>& householderMatrix, const Matrix<float>& vector, int shift)
{
    // Setup GPU for cuBLAS and compute the new colum/row pair
    int dim = vector.GetDimY();
    float alpha = 1.0f;
    float beta = 0.0f;
    float *cuHouseholderMatrix, *cuInputVector, *cuOutputVector;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    // Allocate memory
    float *outputVector = new float[dim];
    cudaMalloc(&cuHouseholderMatrix, (size_t)dim*dim*sizeof(*cuHouseholderMatrix)); CUDA_CHECK;
    cudaMalloc(&cuInputVector, (size_t)dim*sizeof(*cuInputVector)); CUDA_CHECK;
    cudaMalloc(&cuOutputVector, (size_t)dim*sizeof(*cuOutputVector)); CUDA_CHECK;
   
    // Create cuBLAS handle
    stat = cublasCreate(&handle);
    
    // Initialize matrix and vector
    stat = cublasSetMatrix(dim, dim, sizeof(*householderMatrix.GetMatrixData()), householderMatrix.GetMatrixData(), dim, cuHouseholderMatrix, dim);
    stat = cublasSetVector(dim, sizeof(*vector.GetMatrixData()), vector.GetMatrixData(), 1, cuInputVector, 1);

    // Perform block pair computations
    stat = cublasSgemv(handle, CUBLAS_OP_N, dim, dim, &alpha, cuHouseholderMatrix, dim, cuInputVector, 1, &beta, cuOutputVector, 1);
    
    // Retreive block pair matrix
    stat = cublasGetVector(dim, sizeof(float), cuOutputVector, 1, outputVector, 1);

    // Copy over new row/column pair elements
    for(int i = 0; i < dim; i++)
    {
        float element = outputVector[IDX2C(0, i, 1)];
     
        // Assign the new matrix element if greater than some zero tolerance
        if(fabs(element) > ZERO_TOL)
        {
            targetMatrix.SetElement(shift, i + shift + 1, element);
            targetMatrix.SetElement(i + shift + 1, shift, element);
            // targetMatrix.SetElement(0, i+1, element);
            // targetMatrix.SetElement(i+1, 0, element);  
        }
        else
        {
            targetMatrix.SetElement(shift, i + shift + 1, 0.0);
            targetMatrix.SetElement(i + shift + 1, shift, 0.0);
            // targetMatrix.SetElement(0, i+1, 0.0);
            // targetMatrix.SetElement(i+1, 0, 0.0);
        }
    }

    // Free memory
    cudaFree(cuHouseholderMatrix);
    cudaFree(cuInputVector);
    cudaFree(cuOutputVector);
    delete outputVector;
    cublasDestroy(handle);
}

template <>
void MatrixNumerics<float>::UpdateMatrixBlock(Matrix<float>& targetMatrix, const Matrix<float>& householderMatrix, int x, int y, int xDim, int yDim)
{
    // Setup GPU for cuBLAS and compute the new colum/row pair
    int dim = householderMatrix.GetDimY();
    float alpha = 1.0f;
    float beta = 0.0f;
    float *cuHouseholderMatrix, *cuAMatrix, *cuTempMatrix;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    // Allocate memory
    float *aMatrixData = targetMatrix.GetBlockData(x, y, xDim, yDim);
    cudaMalloc(&cuHouseholderMatrix, (size_t)dim*dim*sizeof(*cuHouseholderMatrix)); CUDA_CHECK;
    cudaMalloc(&cuAMatrix, (size_t)dim*dim*sizeof(*cuAMatrix)); CUDA_CHECK;
    cudaMalloc(&cuTempMatrix, (size_t)dim*dim*sizeof(*cuTempMatrix)); CUDA_CHECK;
   
    // Create cuBLAS handle
    stat = cublasCreate(&handle);
    
    // Initialize matrices
    stat = cublasSetMatrix(dim, dim, sizeof(*householderMatrix.GetMatrixData()), householderMatrix.GetMatrixData(), dim, cuHouseholderMatrix, dim);
    stat = cublasSetMatrix(dim, dim, sizeof(*aMatrixData), aMatrixData, dim, cuAMatrix, dim);

    // Perform matrix block computations
    stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim, &alpha, cuHouseholderMatrix, dim, cuAMatrix, dim, &beta, cuTempMatrix, dim);
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, cuTempMatrix, dim, cuHouseholderMatrix, dim, &beta, cuAMatrix, dim);

    // Retreive computed block matrix
    stat = cublasGetMatrix(dim, dim, sizeof(*aMatrixData), cuAMatrix, dim, aMatrixData, dim);

    // Copy new computed elements to target matrix
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            targetMatrix.SetElement(x + j, y + i, aMatrixData[IDX2C(i, j, dim)]);
        }  
    }

    // Free memory
    cudaFree(cuHouseholderMatrix);
    cudaFree(cuAMatrix);
    cudaFree(cuTempMatrix);
    delete aMatrixData;
    cublasDestroy(handle);
}

template <>
void MatrixNumerics<float>::UpdateMatrixBlockDiagonals(Matrix<float>& targetMatrix, Matrix<float>& householderMatrix, int x, int y, int xDim, int yDim)
{
    // Setup GPU for cuBLAS and compute the new colum/row pair
    int dim = householderMatrix.GetDimY();
    float alpha = 1.0f;
    float beta = 0.0f;
    float *cuHouseholderMatrix, *cuOldAMatrix, *cuNewAMatrix;
    cublasStatus_t stat;
    cublasHandle_t handle;

    float *aMatrixData = new float[dim*dim];
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            aMatrixData[IDX2C(i, j, dim)] = targetMatrix.GetElement(x + j, y + i);
        }
    }

    // Allocate memory
    cudaMalloc(&cuHouseholderMatrix, (size_t)dim*dim*sizeof(*cuHouseholderMatrix)); CUDA_CHECK;
    cudaMalloc(&cuOldAMatrix, (size_t)dim*dim*sizeof(*cuOldAMatrix)); CUDA_CHECK;
    cudaMalloc(&cuNewAMatrix, (size_t)dim*dim*sizeof(*cuNewAMatrix)); CUDA_CHECK;
   
    // Create cuBLAS handle
    stat = cublasCreate(&handle);
    
    // Initialize matrices
    stat = cublasSetMatrix(dim, dim, sizeof(*householderMatrix.GetMatrixData()), householderMatrix.GetMatrixData(), dim, cuHouseholderMatrix, dim);
    stat = cublasSetMatrix(dim, dim, sizeof(*aMatrixData), aMatrixData, dim, cuOldAMatrix, dim);
    stat = cublasSetMatrix(dim, dim, sizeof(*aMatrixData), aMatrixData, dim, cuOldAMatrix, dim);

    // Compute new matrix that needs first column minimized by a new householder matrix
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, cuOldAMatrix, dim, cuHouseholderMatrix, dim, &beta, cuNewAMatrix, dim);

    // Retreive new block diagonal matrix
    stat = cublasGetMatrix(dim, dim, sizeof(*aMatrixData), cuNewAMatrix, dim, aMatrixData, dim);

    // Retrieve first column of new block diagonal
    Matrix<float> *aMatrixColumn = new Matrix<float>(1, dim, 0.0);
    for(int i = 0; i < aMatrixColumn->GetDimY(); i++)
    {
        aMatrixColumn->SetElement(0, i, aMatrixData[IDX2C(i, 0, dim)]);
    }
    // aMatrixColumn->Print();

    // Compute the new householder matrix
    Matrix<float> *newHouseholderMatrix = MatrixNumerics<float>::GetRestrictedHouseholderMatrix(*aMatrixColumn, true);

    // Set GPU householder matrix to updated version
    stat = cublasSetMatrix(dim, dim, sizeof(*newHouseholderMatrix->GetMatrixData()), newHouseholderMatrix->GetMatrixData(), dim, cuHouseholderMatrix, dim);

    // Update block diagonal elements in target matrix
    stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim, &alpha, cuHouseholderMatrix, dim, cuNewAMatrix, dim, &beta, cuOldAMatrix, dim);

    // Retrieve new block diagonal elements and householder matrix
    stat = cublasGetMatrix(dim, dim, sizeof(*aMatrixData), cuOldAMatrix, dim, aMatrixData, dim);
    stat = cublasGetMatrix(dim, dim, sizeof(*householderMatrix.GetMatrixData()), cuHouseholderMatrix, dim, householderMatrix.GetMatrixData(), dim);

    // Copy new computed elements to target matrix (required due to matrix indexing differences)
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < dim; j++)
        {
            float element = aMatrixData[IDX2C(i, j, dim)];
         
            // Assign the new matrix element if greater than some zero tolerance
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

    // Free memory
    cudaFree(cuHouseholderMatrix);
    cudaFree(cuOldAMatrix);
    cudaFree(cuNewAMatrix);
    delete aMatrixData, aMatrixColumn, newHouseholderMatrix;
    cublasDestroy(handle);
}

template <class T>
Matrix<T>* MatrixNumerics<T>::LangTridiagonalization21(const Toeplitz<T>& matrix)
{
    Toeplitz<T> *outputMatrix;

    // Only run against square matrices
    if(matrix.GetDimX() == matrix.GetDimY())
    {
        int x, y;  // Current top-left coordinates
        int baseBlocks, remBlocks;
        int dim = matrix.GetDimX();  // Dimension of matrix
        int bandwidth = matrix.GetBandwidth();  // Bandwidth of square, banded matrix
        outputMatrix = new Toeplitz<T>(dim, bandwidth, matrix.GetMatrixData());

        std::cout << "dimension = " << dim << std::endl;
        std::cout << "bandwidth = " << bandwidth << std::endl;

        // Main loop for transforming matrix into tridiagonal form
        // for(int nu = 0; nu < 3; nu++)
        for(int nu = 0; nu < dim - 2; nu++)
        {
            // Set the top-left origin coordinates
            x = nu;
            y = nu;

            std::cout << "x = " << x << std::endl;
            std::cout << "y = " << y << std::endl;

            // Determine the b & r iteration parameters
            baseBlocks = (dim - nu - 1) / bandwidth;
            remBlocks = dim - nu - bandwidth*(baseBlocks - 1) - 1;
            std::cout << "baseBlocks = " << baseBlocks << std::endl;
            std::cout << "remBlocks = " << remBlocks << std::endl;
            if(remBlocks > bandwidth)
            {
                baseBlocks += (remBlocks / bandwidth);
                remBlocks -= bandwidth*(remBlocks / bandwidth);
                std::cout << "baseBlocks = " << baseBlocks << std::endl;
                std::cout << "remBlocks = " << remBlocks << std::endl;
            }

            // Compute the Householder matrix
            Matrix<T> *column = outputMatrix->GetBlock(x, y + 1, 1, bandwidth);
            column->Print();
            outputMatrix->Print();
            Matrix<T> *householderMatrix = GetRestrictedHouseholderMatrix(*column, false);

            // Update the colum/row pair of the main block
            MatrixNumerics<T>::UpdateColumnRowPair(*outputMatrix, *householderMatrix, *column, nu);
            outputMatrix->Print();
            householderMatrix->Print();

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

                    std::cout << "betaX = " << betaX << std::endl;
                    std::cout << "betaY = " << betaY << std::endl;
                    std::cout << "betaDimX = " << betaDimX << std::endl;
                    std::cout << "betaDimY = " << betaDimY << std::endl;

                    MatrixNumerics<T>::UpdateMatrixBlock(*outputMatrix, *householderMatrix, betaX, betaY, betaDimX, betaDimY);
                    outputMatrix->Print();
                    householderMatrix->Print();
                }
                else
                {
                    if(baseBlocks == 1)
                    {
                        betaX = x + 1;
                        betaY = y + 1;
                        betaDimX = remBlocks;
                        betaDimY = betaDimX;
                        // betaX = 2*(beta + 1) + nu;
                        // betaY = betaX;
                        // betaDimX = betaX + bandwidth;
                        // betaDimY = betaY + remBlocks;
                    }
                    else
                    {
                        betaX += bandwidth;
                    }

                    if(remBlocks > 1)
                    {
                        std::cout << "betaX = " << betaX << std::endl;
                        std::cout << "betaY = " << betaY << std::endl;
                        std::cout << "betaDimX = " << betaDimX << std::endl;
                        std::cout << "betaDimY = " << betaDimY << std::endl;

                        MatrixNumerics<T>::UpdateMatrixBlock(*outputMatrix, *householderMatrix, betaX, betaY, betaDimX, betaDimY);
                        outputMatrix->Print();
                        householderMatrix->Print();
                    }
                }

                if(beta < baseBlocks - 2)
                {
                    betaY += bandwidth;

                    std::cout << "betaX = " << betaX << std::endl;
                    std::cout << "betaY = " << betaY << std::endl;
                    std::cout << "betaDimX = " << betaDimX << std::endl;
                    std::cout << "betaDimY = " << betaDimY << std::endl;

                    MatrixNumerics<T>::UpdateMatrixBlockDiagonals(*outputMatrix, *householderMatrix, betaX, betaY, betaDimX, betaDimY);
                    outputMatrix->Print();
                    householderMatrix->Print();
                }
                else if(beta == baseBlocks - 2)
                {
                    betaY += bandwidth;
                    // betaDimY = dim - betaY;
                    // betaDimX = bandwidth;
                    betaDimY = remBlocks;

                    if(remBlocks > 1)
                    {
                        std::cout << "1betaX = " << betaX << std::endl;
                        std::cout << "betaY = " << betaY << std::endl;
                        std::cout << "betaDimX = " << betaDimX << std::endl;
                        std::cout << "betaDimY = " << betaDimY << std::endl;

                        MatrixNumerics<T>::UpdateMatrixBlockDiagonals(*outputMatrix, *householderMatrix, betaX, betaY, betaDimX, betaDimY);
                        outputMatrix->Print();
                        householderMatrix->Print();
                    }
                }
            }

            delete column;
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
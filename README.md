atlas
=====
Symmetric Banded Matrix Reduction to Tridiagonal Form via Householder Transformations

The purpose of this project was to tridiagonalizes high-dimensional symmetric-banded matrices via Householder transformations using Nvidia GPUs. The programming languages/technologies used include C++, Boost, STL, CUDA, and cuBLAS.

The algorithms implemented for computing tridiagonal symmetric matrices included the Householder transormation procedure described in two sources: 

1. Numerical Analysis by Burden and Faires
2. “A Parallel Algorithm for Reducing Symmetric Banded Matrices to Triadiagonal Form” by Bruno Lang

The former is the goto method for computing these types of matrices while the latter is a new method that employs similar numerical methods, but in a highly parallelized manner.  These two solutions can be found in the  **householder_algo** and **lang_2_1_algo** directories, respectively. MATLAB scripts to show proof of concepts for both sets of algorithms can be found in the **matlab** directory.

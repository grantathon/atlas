#ifndef TOEPLITZ_H
#define TOEPLITZ_H

/* Builds a Toeplitz matrix */
int BuildToeplitz(float *t_matrix, int dim, int diag_cnt);

/* Builds a symmetric Toeplitz matrix */
int BuildSymmetricToeplitz(float *t_matrix, int dim, int diag_cnt);

#endif

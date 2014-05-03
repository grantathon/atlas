#include "toeplitz.h"
#include <stdio.h>

int BuildToeplitz(float *t_matrix, int dim, int diag_cnt)
{
    // Determine parameter validity
    if(diag_cnt >= 2*dim || diag_cnt == 0 || diag_cnt % 2 == 0)
    {
        printf("The variable diag_cnt must be odd, less than two times dim, and not zero.\n");
        return -1;
    }
    
    float *diagConsts = (float *)malloc(diag_cnt*sizeof(*diagConsts));

    // Assigns random numbers to the diagonal constants
    for(size_t i = 0; i < (size_t)diag_cnt; i++)
    {
        diagConsts[i] = ((float)rand() / (float)RAND_MAX);
    }

    int diagRadius = diag_cnt - (diag_cnt / 2) - 1;

    for(size_t y = 0; y < (size_t)dim; y++)
    {
        for(size_t x = 0; x < (size_t)dim; x++)
        {
            const size_t idx = x + dim*y;
            
            if(abs((int)x-(int)y) > diagRadius)
            {
                t_matrix[idx] = 0.0;
            }
            else
            {
                t_matrix[idx] = diagConsts[diagRadius + (x - y)];
            }
        }
    }
    
    free(diagConsts);
    return 0;
}

void PrintMatrix(float *matrix, int m, int n)
{
    for(unsigned long i = 0; i < (size_t)m; i++)
    {
        printf("|");

        for(unsigned long j = 0; j < (size_t)n; j++)
        {
            const size_t idx = j + m*i;
            printf(" %1.2f", matrix[idx]);
        }

        printf(" |\n");
    }
}


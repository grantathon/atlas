#include "toeplitz.h"
#include "aux.h"
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

    // Assigns "random" numbers to the diagonal constants
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

int BuildSymmetricToeplitz(float *t_matrix, int dim, int r)
{
    size_t idx = 0;
    int absXY = 0;

    // Determine parameter validity
    if(r >= dim)
    {
        printf("The variable r less than dim.\n");
        return -1;
    }
    
    float *diagConsts = (float *)malloc(r*sizeof(*diagConsts));

    // Assigns "random" numbers to the diagonal constants
    for(size_t i = 0; i < (size_t)(r+1); i++)
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
    
    free(diagConsts);
    return 0;
}

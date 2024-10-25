#include"cuda_errchk.h"
#include<stdio.h>
void cuda_error_check(const char* file, const char* function)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error in %s.%s : %s\n", file, function, cudaGetErrorString(err));
}

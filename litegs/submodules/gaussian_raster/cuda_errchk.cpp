#include"cuda_errchk.h"
#include<stdio.h>
void cuda_error_check(const char* file, const char* function)
{
    musaDeviceSynchronize();
    musaError_t err = musaGetLastError();
    if (err != musaSuccess)
        printf("Error in %s.%s : %s\n", file, function, musaGetErrorString(err));
}
